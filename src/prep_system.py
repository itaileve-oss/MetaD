"""Prepare a TNF trimer for MD: clean, protonate, solvate, minimise, equilibrate.

Two systems are needed for the A->B free energy project:
  * state A: apo symmetric trimer, 1TNF
  * state B: the asymmetric trimer with the LIGAND DELETED, 7KP9

State B must keep only protein coordinates — the whole question is whether that
conformation is accessible without the ligand, so the ligand cannot be present.

Run in the `openfe` env (OpenMM 8.4 + pdbfixer + CUDA).

Usage:
    python src/prep_system.py INPUT.pdb OUT_DIR [--keep-chains ABC] [--pad 1.2]
                              [--equil-ns 0.5] [--platform CUDA]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import openmm
import openmm.app as app
from openmm import unit
from pdbfixer import PDBFixer

import equil_protocol


def strip_heterogens(path: Path, keep_chains: str, out: Path) -> dict:
    """Drop every HETATM (ligand, waters, ions) and keep only the wanted chains."""
    kept, dropped_res, chains_seen = [], {}, set()
    for line in path.read_text().splitlines():
        rec = line[:6]
        if rec == "ATOM  ":
            chain = line[21]
            chains_seen.add(chain)
            if chain in keep_chains:
                # Keep only altLoc blank or A, so no duplicate atoms reach the FF.
                if line[16] in " A":
                    kept.append(line)
        elif rec == "HETATM":
            resname = line[17:20].strip()
            dropped_res[resname] = dropped_res.get(resname, 0) + 1
        elif rec in ("TER   ", "END   "):
            continue
    out.write_text("\n".join(kept) + "\nEND\n")
    return {"chains_in_file": sorted(chains_seen), "kept_chains": list(keep_chains),
            "dropped_heteroatoms": dropped_res, "n_atoms_kept": len(kept)}


def remap_pdb(path: Path, out: Path, chain_map: dict[str, str], res_range: tuple[int, int] | None) -> dict:
    """Rename chains (e.g. 7JRA C->A), keep residues inside res_range, write chains in new-ID order.

    SEQRES lines are kept and renamed too: PDBFixer needs them to rebuild loops and
    the terminal residues that make every chain span the same range.
    """
    seqres: dict[str, list[str]] = {}
    atoms: dict[str, list[str]] = {}
    dropped = 0
    for line in path.read_text().splitlines():
        rec = line[:6]
        if rec == "SEQRES" and line[11] in chain_map:
            new = chain_map[line[11]]
            seqres.setdefault(new, []).append(line[:11] + new + line[12:])
        elif rec == "ATOM  " and line[21] in chain_map:
            num = int(line[22:26])
            if res_range and not (res_range[0] <= num <= res_range[1]):
                dropped += 1
                continue
            new = chain_map[line[21]]
            atoms.setdefault(new, []).append(line[:21] + new + line[22:])
    order = sorted(atoms)
    lines = [l for c in order for l in seqres.get(c, [])]
    for c in order:
        lines += atoms[c] + ["TER"]
    out.write_text("\n".join(lines) + "\nEND\n")
    return {"chain_map": chain_map, "res_range": res_range, "atoms_dropped_out_of_range": dropped}


def graft_loops(path: Path, out: Path, donor: str, flank: int = 4) -> dict:
    """Fill internal gaps in every chain with the EXPERIMENTAL loop from chain `donor`.

    Homotrimer: all chains share one sequence, so a loop resolved in the donor chain is
    the best available model for the same loop in a chain where it is disordered.
    PDBFixer's own loop builder places residues randomly and left a badly strained
    loop at chain A 105-111 (|F| ~1e7-3e8 kJ/mol/nm), right next to the S2/S3 groups.
    The donor loop is superposed on `flank` CA atoms on EACH side of the gap.
    """
    import numpy as np
    lines = path.read_text().splitlines()
    res: dict[str, dict[int, list[str]]] = {}
    for l in lines:
        if l.startswith("ATOM  "):
            res.setdefault(l[21], {}).setdefault(int(l[22:26]), []).append(l)

    def ca(chain, num):
        for l in res[chain].get(num, []):
            if l[12:16].strip() == "CA":
                return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
        return None

    report = {"donor": donor, "grafted": {}}
    for ch in sorted(res):
        if ch == donor:
            continue
        nums = sorted(res[ch])
        for a, b in zip(nums, nums[1:]):
            if b - a <= 1:
                continue
            gap = list(range(a + 1, b))
            if not all(n in res[donor] for n in gap):
                raise SystemExit(f"ABORT: donor {donor} lacks residues {gap}")
            fl = [n for n in range(a - flank + 1, a + 1)] + [n for n in range(b, b + flank)]
            P = np.array([ca(donor, n) for n in fl]); Q = np.array([ca(ch, n) for n in fl])
            pc, qc = P.mean(0), Q.mean(0)
            u, _, vt = np.linalg.svd((P - pc).T @ (Q - qc))
            dd = np.sign(np.linalg.det(vt.T @ u.T))
            R = vt.T @ np.diag([1, 1, dd]) @ u.T
            rmsd = float(np.sqrt((((P - pc) @ R.T + qc - Q) ** 2).sum(1).mean()))
            for n in gap:
                new = []
                for l in res[donor][n]:
                    x = np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
                    y = (x - pc) @ R.T + qc
                    new.append(l[:21] + ch + l[22:30] + f"{y[0]:8.3f}{y[1]:8.3f}{y[2]:8.3f}" + l[54:])
                res[ch][n] = new

            def atom(c, n, name):
                for l in res[c][n]:
                    if l[12:16].strip() == name:
                        return np.array([float(l[30:38]), float(l[38:46]), float(l[46:54])])
            j1 = float(np.linalg.norm(atom(ch, a, "C") - atom(ch, a + 1, "N")))
            j2 = float(np.linalg.norm(atom(ch, b - 1, "C") - atom(ch, b, "N")))
            report["grafted"][f"{ch}:{gap[0]}-{gap[-1]}"] = {
                "flank_CA_rmsd_A": round(rmsd, 3), "junction_C-N_A": [round(j1, 2), round(j2, 2)]}
            print(f"grafted {donor}:{gap[0]}-{gap[-1]} -> chain {ch}: flank CA RMSD {rmsd:.2f} A, "
                  f"junction C-N {j1:.2f} / {j2:.2f} A")
            if max(j1, j2) > 2.5:
                raise SystemExit("ABORT: grafted loop junction C-N > 2.5 A")
    keep = [l for l in lines if l.startswith("SEQRES")]
    for ch in sorted(res):
        for n in sorted(res[ch]):
            keep += res[ch][n]
        keep.append("TER")
    out.write_text("\n".join(keep) + "\nEND\n")
    return report


def his_variants_from(ref_pdb: Path) -> list[str | None]:
    """Per-residue Modeller variants copying the reference's histidine protonation."""
    top = app.PDBFile(str(ref_pdb)).topology
    out: list[str | None] = []
    for res in top.residues():
        if res.name in ("HIS", "HID", "HIE", "HIP"):
            names = {a.name for a in res.atoms()}
            d, e = "HD1" in names, "HE2" in names
            out.append("HIP" if d and e else "HID" if d else "HIE")
        else:
            out.append(None)
    return out


def reorder_like(top, pos, ref_top):
    """Rebuild (top, pos) with each residue's atoms in the reference's per-residue order.

    PDBFixer emits rebuilt residues in TEMPLATE order but crystal residues in FILE
    order, so two otherwise identical systems can disagree on atom order - fatal for
    anything that pairs trajectory frames atom-by-atom (Gen-COMPAS, RiteWeight).
    """
    new = app.Topology()
    new.setPeriodicBoxVectors(top.getPeriodicBoxVectors())
    amap, new_pos = {}, []
    ref_res = list(ref_top.residues())
    for res, rres in zip(top.residues(), ref_res):
        if res.chain.index == len(list(new.chains())):
            chain = new.addChain(res.chain.id)
        r = new.addResidue(res.name, chain, res.id, res.insertionCode)
        by_name = {a.name: a for a in res.atoms()}
        order = [a.name for a in rres.atoms()]
        if set(order) != set(by_name):
            raise SystemExit(f"ABORT: atom sets differ at {res.chain.id}{res.name}{res.id}")
        for name in order:
            a = by_name[name]
            amap[a] = new.addAtom(name, a.element, r)
            new_pos.append(pos[a.index])
    for b in top.bonds():
        new.addBond(amap[b[0]], amap[b[1]])
    return new, unit.Quantity([p.value_in_unit(unit.nanometer) for p in new_pos], unit.nanometer)


def main() -> int:
    ap = argparse.ArgumentParser()
    # Matched-topology mode (Gen-COMPAS needs A and B to share ONE protein atom list):
    ap.add_argument("--chain-map", default=None,
                    help="rename chains before fixing, e.g. 'C:A,B:B,A:C' (7JRA onto 1TNF order)")
    ap.add_argument("--res-range", default=None,
                    help="keep only residues lo-hi in the INPUT numbering, and rebuild "
                         "missing terminal residues inside it, e.g. 82-233")
    ap.add_argument("--variants-from", type=Path, default=None,
                    help="copy histidine protonation from this fixed.pdb (residue order must match)")
    ap.add_argument("--mutate", default=None,
                    help="PDBFixer mutation applied to every kept chain, input numbering, "
                         "e.g. LEU-143-ASP (1TNF carries L at ECD 143; wild type is D)")
    ap.add_argument("--renumber-start", type=int, default=None,
                    help="renumber every chain consecutively from this residue number "
                         "(6 = TNF soluble-domain numbering, as in 1TNF). PDBFixer otherwise "
                         "restarts each chain at 1, which hides which residue is which.")
    ap.add_argument("--fix-only", action="store_true",
                    help="stop after writing fixed.pdb (no solvation/MD)")
    ap.add_argument("input_pdb", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--keep-chains", default="ABC")
    ap.add_argument("--pad", type=float, default=1.2, help="solvent padding, nm")
    ap.add_argument("--platform", default="CUDA")
    ap.add_argument("--ph", type=float, default=7.4)
    # Defaults below follow bioRxiv 10.1101/2025.11.28.690881 so our free energies are
    # comparable with theirs: ff14SB + TIP3P + 150 mM KCl, 310 K, 4 fs with hydrogen
    # mass repartitioning, 12 A vdW cutoff switching from 10 A.
    ap.add_argument("--temp", type=float, default=310.0)
    ap.add_argument("--timestep-fs", type=float, default=4.0)
    ap.add_argument("--hmr", type=float, default=1.5, help="hydrogen mass (amu); 0 disables HMR")
    ap.add_argument("--cutoff", type=float, default=1.2, help="nm")
    ap.add_argument("--switch", type=float, default=1.0, help="nm")
    ap.add_argument("--ionic", type=float, default=0.15, help="molar KCl")
    ap.add_argument("--equil-scale", type=float, default=1.0,
                    help="multiply every equilibration stage length (SMOKE TESTS ONLY)")
    ap.add_argument("--graft-loops-from", default=None,
                    help="fill internal gaps with the experimental loop of this chain "
                         "(after --chain-map), e.g. C for 7JRA")
    ap.add_argument("--pocket-ref", type=Path, default=None,
                    help="PDB holding a ligand whose volume should be hydrated (check only)")
    ap.add_argument("--pocket-resname", default="VGY")
    args = ap.parse_args()

    d = args.out_dir
    d.mkdir(parents=True, exist_ok=True)
    report: dict = {"input": str(args.input_pdb), "keep_chains": args.keep_chains}

    # Load the ORIGINAL file so PDBFixer reads its SEQRES records, then let PDBFixer
    # strip heterogens itself.
    #
    # Do NOT pre-strip the file by keeping only ATOM lines: that drops SEQRES, so
    # findMissingResidues() finds nothing, and the gaps get RENUMBERED CONTIGUOUSLY.
    # OpenMM then bonds residues that are 9-12 A apart, and minimisation contracts
    # those fake peptide bonds and drags the structure together — silently destroying
    # the very subunit geometry these CVs measure. That happened with 7KP9 (8 glued
    # gaps, worst CA-CA 11.7 A).
    res_range = tuple(int(x) for x in args.res_range.split("-")) if args.res_range else None
    src = args.input_pdb
    if args.chain_map or res_range:
        cmap = (dict(p.split(":") for p in args.chain_map.split(",")) if args.chain_map
                else {c: c for c in args.keep_chains})
        src = d / "remapped_input.pdb"
        report["remap"] = remap_pdb(args.input_pdb, src, cmap, res_range)
    observed_src = src
    if args.graft_loops_from:
        g = d / "grafted_input.pdb"
        report["loop_graft"] = graft_loops(src, g, args.graft_loops_from)
        src = g
    fixer = PDBFixer(filename=str(src))
    fixer.removeHeterogens(keepWater=False)
    all_chain_ids = [c.id for c in fixer.topology.chains()]
    drop = [i for i, c in enumerate(fixer.topology.chains()) if c.id not in args.keep_chains]
    if drop:
        fixer.removeChains(drop)
    report["chains_in_file"] = all_chain_ids
    report["chains_kept"] = [c.id for c in fixer.topology.chains()]

    if args.mutate:
        for c in [c.id for c in fixer.topology.chains()]:
            fixer.applyMutations([args.mutate], c)
        report["mutation"] = args.mutate

    fixer.findMissingResidues()
    # Rebuild only INTERNAL loops. Missing residues at a chain terminus would be added
    # as long unstructured tails that do nothing but cost sampling time.
    keys = list(fixer.missingResidues)
    chains = list(fixer.topology.chains())
    trimmed = {}
    terminal_added = {}
    for key in keys:
        chain_idx, pos = key
        chain = chains[chain_idx]
        residues = list(chain.residues())
        n_res = len(residues)
        if pos == 0 or pos == n_res:
            if res_range is None:
                continue  # terminal
            # Matched mode: add just enough terminal residues to reach the range ends.
            if pos == 0:
                need = int(residues[0].id) - res_range[0]
                names = fixer.missingResidues[key][-need:] if need > 0 else []
            else:
                need = res_range[1] - int(residues[-1].id)
                names = fixer.missingResidues[key][:need] if need > 0 else []
            if names:
                trimmed[key] = names
                terminal_added[f"{chain.id}:{'N' if pos == 0 else 'C'}"] = names
            continue
        trimmed[key] = fixer.missingResidues[key]
    report["terminal_residues_added"] = terminal_added
    report["missing_residues_terminal_skipped"] = len(keys) - len(trimmed)
    fixer.missingResidues = trimmed
    report["missing_residues"] = {str(k): len(v) for k, v in trimmed.items()}
    report["n_loop_residues_rebuilt"] = sum(len(v) for v in trimmed.values())
    fixer.findNonstandardResidues()
    report["nonstandard"] = [str(r) for r in fixer.nonstandardResidues]
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    report["missing_atoms_n"] = len(fixer.missingAtoms)
    fixer.addMissingAtoms()
    if args.variants_from:
        # Histidine tautomers are chosen from local H-bonding, so two conformations of
        # the same protein can get DIFFERENT atom lists. Copy them from the reference.
        variants = his_variants_from(args.variants_from)
        n_res = fixer.topology.getNumResidues()
        if len(variants) != n_res:
            raise SystemExit(f"ABORT: reference has {len(variants)} residues, this system {n_res}")
        ff_h = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
        mod = app.Modeller(fixer.topology, fixer.positions)
        mod.addHydrogens(ff_h, pH=args.ph, variants=variants)
        fixer.topology, fixer.positions = reorder_like(
            mod.topology, mod.positions, app.PDBFile(str(args.variants_from)).topology)
        report["his_variants_copied_from"] = str(args.variants_from)
    else:
        fixer.addMissingHydrogens(args.ph)
    if args.renumber_start is not None:
        for chain in fixer.topology.chains():
            for i, res in enumerate(chain.residues()):
                res.id = str(args.renumber_start + i)
        report["renumbered_from"] = args.renumber_start
    with (d / "fixed.pdb").open("w") as fh:
        # keepIds: otherwise OpenMM restarts every chain at residue 1 on write.
        app.PDBFile.writeFile(fixer.topology, fixer.positions, fh,
                              keepIds=args.renumber_start is not None)

    # Guard: consecutive CA atoms more than ~4.5 A apart mean an unrebuilt gap that
    # OpenMM will bond anyway. Fail loudly rather than equilibrate a corrupted system.
    import math
    ca = {}
    for res in fixer.topology.residues():
        for atom in res.atoms():
            if atom.name == "CA":
                ca.setdefault(res.chain.id, []).append(
                    (res.index, fixer.positions[atom.index].value_in_unit(unit.nanometer))
                )
    breaks = []
    for cid, lst in ca.items():
        lst.sort()
        for (i, a), (j, b) in zip(lst, lst[1:]):
            dist = 10.0 * math.dist(a, b)  # nm -> A
            if dist > 4.5:
                breaks.append((cid, i, j, round(dist, 2)))
    report["chain_breaks"] = breaks
    if breaks:
        raise SystemExit(
            f"ABORT: {len(breaks)} unrebuilt chain break(s) {breaks}. "
            "SEQRES is probably missing from the input, so loops were not rebuilt."
        )
    print(f"chain-break check passed ({sum(len(v) for v in ca.values())} CA)")
    print(f"fixed: {fixer.topology.getNumResidues()} residues, "
          f"{fixer.topology.getNumAtoms()} atoms; missing loops {report['missing_residues']}")
    if args.variants_from:
        ref_top = app.PDBFile(str(args.variants_from)).topology
        key = lambda t: [(a.residue.chain.index, a.residue.name, a.name) for a in t.atoms()]
        same = key(ref_top) == key(fixer.topology)
        report["protein_atoms_identical_to_reference"] = same
        print(f"protein atom list identical to {args.variants_from}: {same}")
        if not same:
            (d / "prep_report.json").write_text(json.dumps(report, indent=2))
            raise SystemExit("ABORT: protein topology differs from the reference")
    if args.fix_only:
        (d / "prep_report.json").write_text(json.dumps(report, indent=2))
        return 0

    # amber14-all.xml IS ff14SB; plain tip3p (not tip3pfb) to match the paper.
    # (Checked 2026-11-07: addSolvent already ignores eps=0 atoms such as the ff14SB
    # hydroxyl H, so no solvation workaround is needed.)
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    modeller = app.Modeller(fixer.topology, fixer.positions)
    # Atoms NOT observed in the input (grafted/rebuilt loops, added termini, side-chain
    # atoms PDBFixer added, e.g. 7JRA Glu146/Asn92): their position matches no atom of
    # the pre-graft input. They are left unrestrained during equilibration, so only
    # crystallographically observed heavy atoms are held (PDBFixer can place added
    # atoms in clashes, |F| ~1e8, which restraints would otherwise pin in place).
    import numpy as np
    from scipy.spatial import cKDTree
    obs = np.array([[float(l[30:38]), float(l[38:46]), float(l[46:54])]
                    for l in Path(observed_src).read_text().splitlines()
                    if l.startswith(("ATOM", "HETATM"))])
    pos_A = np.array(fixer.positions.value_in_unit(unit.angstrom))
    dist, _ = cKDTree(obs).query(pos_A)
    unobserved = {int(i) for i in np.nonzero(dist > 0.01)[0]}
    atoms_list = list(fixer.topology.atoms())
    heavy_unobs = [i for i in unobserved if atoms_list[i].element.symbol != "H"]
    mod_res = sorted({(atoms_list[i].residue.chain.id, int(atoms_list[i].residue.id),
                       atoms_list[i].residue.name) for i in heavy_unobs})
    report["unobserved_heavy_atoms"] = len(heavy_unobs)
    report["residues_with_unobserved_heavy_atoms"] = [f"{c}:{n}{r}" for c, r, n in mod_res]
    print(f"unobserved heavy atoms (unrestrained): {len(heavy_unobs)} in "
          f"{report['residues_with_unobserved_heavy_atoms']}")
    modeller.addSolvent(ff, model="tip3p", padding=args.pad * unit.nanometer,
                        positiveIon="K+", negativeIon="Cl-",
                        ionicStrength=args.ionic * unit.molar, neutralize=True)
    report["n_atoms_solvated"] = modeller.topology.getNumAtoms()
    print(f"solvated: {report['n_atoms_solvated']} atoms")

    # HMR lets a 4 fs timestep be stable; it perturbs fast dynamics but not the
    # thermodynamics we are after, and the reference paper used it.
    hmass = (args.hmr * unit.amu) if args.hmr else None
    system = ff.createSystem(modeller.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=args.cutoff * unit.nanometer,
                             switchDistance=args.switch * unit.nanometer,
                             constraints=app.HBonds, rigidWater=True,
                             hydrogenMass=hmass, ewaldErrorTolerance=0.0005)
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, args.temp * unit.kelvin, 25))
    # The production System (no restraints): this is what system.xml stores.
    system_xml = openmm.XmlSerializer.serialize(system)
    report.update({"temp_K": args.temp, "timestep_fs": args.timestep_fs,
                   "hydrogen_mass_amu": args.hmr, "cutoff_nm": args.cutoff,
                   "switch_nm": args.switch, "ewald_tol": 0.0005})

    # The CUDA failure mode that matters is not a missing platform but a PTX/driver
    # mismatch, raised when a Context is built. Probe with a throwaway Context.
    platform = None
    for name in (args.platform, "CPU"):
        try:
            cand = openmm.Platform.getPlatformByName(name)
            tmp = openmm.XmlSerializer.deserialize(system_xml)
            openmm.Context(tmp, openmm.VerletIntegrator(0.001), cand)
            platform = cand
            break
        except Exception as exc:  # noqa: BLE001
            print(f"platform {name} unusable: {type(exc).__name__}: {exc}")
    if platform is None:
        raise SystemExit("no usable OpenMM platform")
    report["platform"] = platform.getName()

    eq = equil_protocol.run(d, modeller.topology, modeller.positions, system_xml,
                            platform, args.temp, pocket_ref=args.pocket_ref,
                            pocket_resname=args.pocket_resname, scale=args.equil_scale,
                            unrestrained_atoms=unobserved)
    sim = eq.pop("_sim")
    report["equilibration"] = eq
    print(f"equilibrated {eq['equil_ns_total']} ns in {eq['equil_wall_s']} s "
          f"-> {eq['ns_per_day']} ns/day on {platform.getName()}; final {eq['final']}")

    sim.saveState(str(d / "equilibrated.xml"))
    # The exact System (force field, cutoff, HMR, barostat) so downstream runs never
    # rebuild it with slightly different settings.
    (d / "system.xml").write_text(system_xml)
    # Protein-only copy that KEEPS the residue numbering (the full-system PDB below
    # restarts chains at 1); protein atoms are the first atoms of the system.
    st = sim.context.getState(getPositions=True)
    prot = app.Modeller(sim.topology, st.getPositions())
    prot.delete([r for r in prot.topology.residues()
                 if r.name in ("HOH", "K", "CL", "NA", "Cl-", "K+")])
    with (d / "equilibrated_protein.pdb").open("w") as fh:
        app.PDBFile.writeFile(prot.topology, prot.positions, fh,
                              keepIds=args.renumber_start is not None)
    with (d / "equilibrated.pdb").open("w") as fh:
        app.PDBFile.writeFile(sim.topology,
                              sim.context.getState(getPositions=True).getPositions(),
                              fh)
    (d / "prep_report.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {d}/prep_report.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
