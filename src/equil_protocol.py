"""Staged, restrained equilibration (~9 ns) run before ANY production MD.

Order (AMBER-style force constants, quoted in kcal/mol/A^2; 1 = 418.4 kJ/mol/nm^2):

  0  minimise         protein heavy atoms restrained 10   -> solvent/H relax first
  1  heat  NVT 2 fs   0 -> T in 10 K steps over 1 ns, heavy atoms 10
  2  water NPT 4 fs   2 ns at T, heavy atoms 10           -> waters find their place
                      (density, ions, and any water that can reach buried cavities)
  3  release NPT      1 ns each: (bb, sc) = (5,5) (2,1) (1,0.5) (0.5,0.1) (0.1,0)
  4  free    NPT      1 ns, no restraints, in a fresh Context built from the
                      production System, so the saved state carries no restraint
                      force or parameters.

Restraint reference = the INPUT (crystal/fixed) coordinates, so the restrained stages
hold the starting conformation rather than whatever minimisation produced. Only
OBSERVED heavy atoms are restrained; modelled atoms (grafted/rebuilt loops, added
termini, PDBFixer-added side-chain atoms) are free from the start so they can relax.

Every `log_ps` the protocol logs the C->A CVs (S1/S2/S3, A), theta-spread, the
number of water O inside an optional reference-ligand volume (crystal frame,
CA-superposed), box edge and potential energy -> <out>/equil_log.tsv, and writes a
protein-only DCD of the whole equilibration (equil.dcd) so any change can be dated.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import openmm
import openmm.app as app
from openmm import unit

KCAL_A2 = 418.4  # kcal/mol/A^2 -> kJ/mol/nm^2

# (name, ensemble, ns, timestep_fs, k_backbone, k_sidechain) in kcal/mol/A^2
STAGES = [
    ("heat", "NVT", 1.0, 2.0, 10.0, 10.0),
    ("water", "NPT", 2.0, 4.0, 10.0, 10.0),
    ("rel1", "NPT", 1.0, 4.0, 5.0, 5.0),
    ("rel2", "NPT", 1.0, 4.0, 2.0, 1.0),
    ("rel3", "NPT", 1.0, 4.0, 1.0, 0.5),
    ("rel4", "NPT", 1.0, 4.0, 0.5, 0.1),
    ("rel5", "NPT", 1.0, 4.0, 0.1, 0.0),
    ("free", "NPT", 1.0, 4.0, 0.0, 0.0),
]
HEAT_STEP_K = 10.0
SOLVENT = {"HOH", "K", "CL", "NA", "K+", "Cl-", "Na+"}

CVS = {  # paper CVs, full-length numbering; ECD = paper - 76 (see make_plumed.py)
    "S1": ([130, 131, 132, 200, 201, 202], [130, 131, 132, 200, 201, 202]),
    "S2": ([169, 170, 171, 172, 173, 174, 193, 194, 195, 196, 197, 198],
           [136, 137, 138, 139, 140, 192, 193, 194, 195, 226, 227, 228]),
    "S3": ([151, 152, 153, 173, 174, 175], [140, 141, 142, 190, 191, 192]),
}


class Monitor:
    def __init__(self, topology, ref_pos_A, pair=("C", "A"), ecd_offset=76,
                 pocket_ref: Path | None = None, pocket_resname="VGY"):
        ca = {(r.chain.id, int(r.id)): a.index for r in topology.residues()
              for a in r.atoms() if a.name == "CA" and r.name not in SOLVENT}
        p, q = pair
        self.groups = {cv: ([ca[(p, r - ecd_offset)] for r in ga],
                            [ca[(q, r - ecd_offset)] for r in gc])
                       for cv, (ga, gc) in CVS.items()}
        common = sorted(set.intersection(*[{r for (c, r) in ca if c == ch} for ch in "ABC"]))
        self.sel = {ch: np.array([ca[(ch, r)] for r in common]) for ch in "ABC"}
        self.ca_all = np.array(sorted(ca.values()))
        self.ref_ca = ref_pos_A[self.ca_all]
        self.ow = np.array([a.index for a in topology.atoms()
                            if a.residue.name == "HOH" and a.element is not None
                            and a.element.symbol == "O"])
        self.lig = None
        if pocket_ref is not None:
            self.lig = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                                 for l in Path(pocket_ref).read_text().splitlines()
                                 if l.startswith("HETATM") and l[17:20] == pocket_resname
                                 and l[76:78].strip() != "H"])

    @staticmethod
    def _kabsch(P, Q):
        """R, t such that P @ R.T + t ~ Q."""
        pc, qc = P.mean(0), Q.mean(0)
        u, _, vt = np.linalg.svd((P - pc).T @ (Q - qc))
        dd = np.sign(np.linalg.det(vt.T @ u.T))
        R = vt.T @ np.diag([1, 1, dd]) @ u.T
        return R, qc - pc @ R.T

    def measure(self, X, box_A):
        """X: positions in A, NOT wrapped (molecules continuous)."""
        out = {}
        for cv, (a, c) in self.groups.items():
            out[cv] = float(np.linalg.norm(X[a].mean(0) - X[c].mean(0)))
        angs = []
        for p, q in (("A", "B"), ("B", "C"), ("C", "A")):
            R, _ = self._kabsch(X[self.sel[p]], X[self.sel[q]])
            angs.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        out["theta_spread"] = float(np.std(angs))
        if self.lig is not None:
            R, t = self._kabsch(X[self.ca_all], self.ref_ca)
            centre = X[self.ca_all].mean(0)
            W = X[self.ow]
            W = W - box_A * np.round((W - centre) / box_A)  # nearest image to protein
            W = W @ R.T + t
            dmin = np.linalg.norm(W[:, None] - self.lig[None], axis=2).min(1)
            out["pocket_waters"] = int((dmin < 2.0).sum())
        return out


def _restraint_force(topology, ref_pos_nm, skip_atoms=frozenset()):
    f = openmm.CustomExternalForce(
        "0.5*(isbb*kbb + (1-isbb)*ksc)*periodicdistance(x, y, z, x0, y0, z0)^2")
    f.addGlobalParameter("kbb", 0.0)
    f.addGlobalParameter("ksc", 0.0)
    for p in ("isbb", "x0", "y0", "z0"):
        f.addPerParticleParameter(p)
    n = 0
    for atom in topology.atoms():
        if atom.residue.name in SOLVENT or atom.element is None or atom.element.symbol == "H" \
                or atom.index in skip_atoms:
            continue
        f.addParticle(atom.index, [1.0 if atom.name in ("N", "CA", "C", "O") else 0.0,
                                   *ref_pos_nm[atom.index]])
        n += 1
    return f, n


def run(out_dir: Path, topology, positions, system_xml: str, platform, temp_K: float,
        pocket_ref: Path | None = None, pocket_resname: str = "VGY",
        log_ps: float = 100.0, dcd_ps: float = 20.0, scale: float = 1.0,
        unrestrained_atoms=frozenset()) -> dict:
    """Run the staged protocol. `scale` shortens every stage (smoke tests only).

    Returns the report dict; leaves equilibrated.xml/equilibrated*.pdb to the caller
    via the returned Simulation in report['_sim'].
    """
    import mdtraj.reporters as mdr

    d = Path(out_dir)
    pos_nm = np.array(positions.value_in_unit(unit.nanometer))
    mon = Monitor(topology, pos_nm * 10.0, pocket_ref=pocket_ref,
                  pocket_resname=pocket_resname)
    protein_idx = [a.index for a in topology.atoms() if a.residue.name not in SOLVENT]

    system = openmm.XmlSerializer.deserialize(system_xml)
    baro = [f for f in system.getForces() if isinstance(f, openmm.MonteCarloBarostat)][0]
    baro.setFrequency(0)
    restr, n_restr = _restraint_force(topology, pos_nm, unrestrained_atoms)
    system.addForce(restr)
    integ = openmm.LangevinMiddleIntegrator(temp_K * unit.kelvin, 1 / unit.picosecond,
                                            2.0 * unit.femtosecond)
    sim = app.Simulation(topology, system, integ, platform)
    sim.context.setPositions(positions)

    log = (d / "equil_log.tsv").open("w")
    cols = ["t_ns", "stage", "T_K", "kbb", "ksc", "S1", "S2", "S3", "theta_spread",
            "pocket_waters", "box_nm", "Epot_kJ"]
    log.write("\t".join(cols) + "\n")
    t_ns = 0.0
    T_now = 0.0
    k_now = (10.0, 10.0)

    def record(stage):
        st = sim.context.getState(getPositions=True, getEnergy=True, enforcePeriodicBox=False)
        X = st.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        box = np.array([v[i] for i, v in enumerate(
            st.getPeriodicBoxVectors().value_in_unit(unit.angstrom))])
        m = mon.measure(X, box)
        row = [f"{t_ns:.3f}", stage, f"{T_now:.0f}", f"{k_now[0]:g}", f"{k_now[1]:g}",
               f"{m['S1']:.2f}", f"{m['S2']:.2f}", f"{m['S3']:.2f}",
               f"{m['theta_spread']:.2f}", str(m.get("pocket_waters", "")),
               f"{box[0] / 10:.3f}",
               f"{st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.0f}"]
        log.write("\t".join(row) + "\n")
        log.flush()
        print("  " + "  ".join(f"{c}={v}" for c, v in zip(cols, row)), flush=True)
        return m

    def set_k(kbb, ksc):
        nonlocal k_now
        k_now = (kbb, ksc)
        sim.context.setParameter("kbb", kbb * KCAL_A2)
        sim.context.setParameter("ksc", ksc * KCAL_A2)

    report = {"protocol": [list(s) for s in STAGES], "n_restrained_atoms": n_restr,
              "restraint_reference": "input (fixed/crystal) coordinates",
              "heat_step_K": HEAT_STEP_K, "scale": scale}
    t0 = time.time()

    # 0) restrained minimisation
    set_k(10.0, 10.0)
    record("input")
    sim.minimizeEnergy()
    record("minimised")
    with (d / "minimised.pdb").open("w") as fh:
        app.PDBFile.writeFile(topology, sim.context.getState(getPositions=True).getPositions(), fh)

    sim.reporters.append(mdr.DCDReporter(str(d / "equil.dcd"), 1, atomSubset=protein_idx))
    dcd = sim.reporters[-1]

    def advance(ns, dt_fs, stage):
        nonlocal t_ns
        n_total = max(1, int(round(ns * scale * 1e6 / dt_fs)))
        chunk = max(1, int(round(log_ps * 1e3 / dt_fs)))
        dcd._reportInterval = max(1, int(round(dcd_ps * 1e3 / dt_fs)))
        done = 0
        while done < n_total:
            n = min(chunk, n_total - done)
            sim.step(n)
            done += n
            t_ns += n * dt_fs * 1e-6
            record(stage)

    for name, ens, ns, dt, kbb, ksc in STAGES:
        if name == "free":
            break
        integ.setStepSize(dt * unit.femtosecond)
        set_k(kbb, ksc)
        if name == "heat":
            sim.context.setVelocitiesToTemperature(1.0 * unit.kelvin)
            n_inc = int(round(temp_K / HEAT_STEP_K))
            for i in range(1, n_inc + 1):
                T_now = min(temp_K, i * HEAT_STEP_K)
                integ.setTemperature(T_now * unit.kelvin)
                advance(ns / n_inc, dt, name)
            T_now = temp_K
            integ.setTemperature(temp_K * unit.kelvin)
            continue
        if ens == "NPT" and baro.getFrequency() == 0:
            baro.setFrequency(25)
            sim.context.reinitialize(preserveState=True)
        advance(ns, dt, name)

    # 4) free NPT in a fresh Context built from the untouched production System
    last = sim.context.getState(getPositions=True, getVelocities=True, enforcePeriodicBox=False)
    name, ens, ns, dt, _, _ = STAGES[-1]
    system = openmm.XmlSerializer.deserialize(system_xml)
    integ = openmm.LangevinMiddleIntegrator(temp_K * unit.kelvin, 1 / unit.picosecond,
                                            dt * unit.femtosecond)
    sim = app.Simulation(topology, system, integ, platform)
    sim.context.setPeriodicBoxVectors(*last.getPeriodicBoxVectors())
    sim.context.setPositions(last.getPositions())
    sim.context.setVelocities(last.getVelocities())
    k_now = (0.0, 0.0)
    sim.reporters.append(mdr.DCDReporter(str(d / "equil_free.dcd"), 1, atomSubset=protein_idx))
    dcd = sim.reporters[-1]
    advance(ns, dt, name)
    log.close()

    total = sum(s[2] for s in STAGES) * scale
    wall = time.time() - t0
    report.update({"equil_ns_total": round(total, 3), "equil_wall_s": round(wall, 1),
                   "ns_per_day": round(total / (wall / 86400), 1),
                   "final": mon.measure(
                       sim.context.getState(getPositions=True, enforcePeriodicBox=False)
                       .getPositions(asNumpy=True).value_in_unit(unit.angstrom),
                       np.array([v[i] for i, v in enumerate(
                           sim.context.getState().getPeriodicBoxVectors()
                           .value_in_unit(unit.angstrom))]))})
    report["_sim"] = sim
    return report
