"""Build and VERIFY the PLUMED input for the paper's three TNF collective variables.

CVs are from bioRxiv 10.1101/2025.11.28.690881: three CoM-to-CoM distances between Ca
subsets on two subunits, in FULL-LENGTH TNF precursor numbering.

Two numbering traps this handles explicitly:
  * 1TNF is deposited with ECD numbering 6-157; 7JRA with full-length 81-233. The
    paper's numbers apply directly to 7JRA but need -76 for 1TNF.
  * pdbfixer RENUMBERS each chain from 1, so the prepped topology needs a per-chain
    offset = (first observed residue in the ORIGINAL pdb) - 1.

Rather than trust that arithmetic, the script recomputes the CVs from the prepped
coordinates and compares against the reference values measured on the crystals
(1TNF symmetric 19.9/8.4/9.3; 7JRA asymmetric 24.2/12.0/14.0). A mismatch aborts.

Usage:
    python src/make_plumed.py RUN_DIR --orig data/1TNF.pdb --ecd-offset 76 \
        --pair A,B --out RUN_DIR/plumed.dat [--expect 19.9,8.4,9.3]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import openmm.app as app
from openmm import unit

# Paper CV definitions, full-length precursor numbering.
# Each entry: (groups on the "A" subunit, groups on the "C" subunit).
CVS = {
    "S1": ([[130, 131, 132], [200, 201, 202]], [[130, 131, 132], [200, 201, 202]]),
    "S2": ([[169, 170, 171, 172, 173, 174], [193, 194, 195, 196, 197, 198]],
           [[136, 137, 138, 139, 140], [192, 193, 194, 195], [226, 227, 228]]),
    "S3": ([[151, 152, 153], [173, 174, 175]], [[140, 141, 142], [190, 191, 192]]),
}
# Paper's sampled ranges (Angstrom) -> used for the restraining walls.
# v2 (2026-11-07): S1 upper 26 -> 28 (v2 basin B sits at S1 ~24-25) and S3 9-15 -> 8-18
# (the paper's 9.0 lower wall cut through basin A at S3 ~9.0-9.5; the v1 run pressed the
# 15 upper wall). The paper's own sampled ranges were S1 [17,26], S2 [7,15], S3 [9,15].
RANGES = {"S1": (17.0, 28.0), "S2": (7.0, 15.0), "S3": (8.0, 18.0)}


def first_resnum_per_chain(pdb_path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in pdb_path.read_text().splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in " A":
            ch, num = line[21], int(line[22:26])
            if ch not in out:
                out[ch] = num
    return out


def ca_index_map(topology) -> dict[tuple[str, int], int]:
    """(chain id, residue number in prepped topology) -> 0-based CA atom index."""
    out = {}
    for res in topology.residues():
        for atom in res.atoms():
            if atom.name == "CA":
                out[(res.chain.id, int(res.id))] = atom.index
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--orig", type=Path, required=True, help="original deposited PDB")
    ap.add_argument("--ecd-offset", type=int, required=True,
                    help="subtract from paper numbering to reach this PDB's numbering "
                         "(76 for 1TNF, 0 for 7JRA)")
    ap.add_argument("--pair", required=True, help="chain pair as A,C")
    ap.add_argument("--expect", default=None, help="expected S1,S2,S3 in A, for validation")
    ap.add_argument("--tol", type=float, default=1.0, help="allowed deviation, A")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--state-pdb", default="equilibrated.pdb",
                    help="structure in RUN_DIR to resolve atom indices against; both "
                         "phases live in one directory so each needs its own")
    # Metadynamics parameters. The paper did not report hills; sigma should come from
    # the unbiased CV fluctuations, so it is an explicit input rather than a guess.
    ap.add_argument("--sigma", default="0.2,0.15,0.15", help="nm, per CV")
    ap.add_argument("--height", type=float, default=1.2, help="kJ/mol")
    ap.add_argument("--biasfactor", type=float, default=15.0)
    ap.add_argument("--pace", type=int, default=1000)
    ap.add_argument("--walkers", type=int, default=4)
    ap.add_argument("--walker-id", type=int, default=0)
    ap.add_argument("--walkers-dir", default="../walkers_shared")
    # 2500 steps x 4 fs = 10 ps. At 500 steps (2 ps) a 250 ns walker writes 125k lines
    # for no analytical benefit.
    ap.add_argument("--colvar-stride", type=int, default=2500)
    # Which CVs to BIAS. All three are always PRINTed so the unbiased one can still be
    # analysed/reweighted. S1+S3 chosen on measured grounds: S3 separates the simulated
    # basins best (5.7 sigma vs S1 4.4, S2 3.1) and is nearly orthogonal to S1
    # (r = +0.03 in A, +0.26 in B), whereas S2 is partly redundant with S1 (r = +0.69 in B).
    ap.add_argument("--bias-cvs", default="S1,S3")
    ap.add_argument("--restart", action="store_true",
                    help="emit PLUMED RESTART so an existing HILLS is read and appended "
                         "to, rather than overwritten; required when resuming or when "
                         "seeding a second phase from the other basin")
    args = ap.parse_args()

    chain_a, chain_c = [c.strip() for c in args.pair.split(",")]
    pdb = app.PDBFile(str(args.run_dir / args.state_pdb))
    idx = ca_index_map(pdb.topology)
    pos = np.array(pdb.positions.value_in_unit(unit.nanometer)) * 10.0  # -> Angstrom

    firsts = first_resnum_per_chain(args.orig)
    # prepped residue number = original - (first_original - 1)
    offsets = {ch: firsts[ch] - 1 for ch in firsts}

    groups: dict[str, dict[str, list[int]]] = {}
    report: dict = {"offsets_original_to_prepped": offsets,
                    "ecd_offset": args.ecd_offset, "pair": [chain_a, chain_c]}
    values = {}
    for cv, (ga, gc) in CVS.items():
        side = {}
        for tag, chain, grouplist in (("a", chain_a, ga), ("c", chain_c, gc)):
            want, got = 0, []
            for grp in grouplist:
                for paper_res in grp:
                    want += 1
                    orig = paper_res - args.ecd_offset
                    prepped = orig - offsets.get(chain, 0)
                    key = (chain, prepped)
                    if key in idx:
                        got.append(idx[key])
            if not got:
                raise SystemExit(f"{cv}: no atoms resolved for chain {chain}")
            if len(got) != want:
                print(f"WARNING {cv} chain {chain}: resolved {len(got)}/{want} residues")
            side[tag] = got
        groups[cv] = side
        values[cv] = float(np.linalg.norm(pos[side["a"]].mean(0) - pos[side["c"]].mean(0)))

    report["cv_values_from_prepped_A"] = {k: round(v, 2) for k, v in values.items()}
    print("CVs recomputed from the prepped structure (A):",
          {k: round(v, 2) for k, v in values.items()})

    if args.expect:
        exp = [float(x) for x in args.expect.split(",")]
        bad = [(cv, values[cv], e) for cv, e in zip(CVS, exp)
               if abs(values[cv] - e) > args.tol]
        if bad:
            raise SystemExit(
                "ABORT: CV mismatch vs crystal reference "
                + "; ".join(f"{cv}: got {g:.2f}, expected {e:.2f}" for cv, g, e in bad)
                + ". Numbering/offset or chain pair is wrong."
            )
        print(f"validation passed (within {args.tol} A of {exp})")

    solvent = {"HOH", "WAT", "NA", "CL", "K", "MG", "ZN"}
    protein_chain_ranges = []
    for chain in pdb.topology.chains():
        idxs = [a.index + 1 for r in chain.residues()
                if r.name.strip() not in solvent for a in r.atoms()]
        if idxs:
            protein_chain_ranges.append((min(idxs), max(idxs)))
    if not protein_chain_ranges:
        raise SystemExit("no protein chains found for WHOLEMOLECULES")
    report["protein_chain_ranges"] = protein_chain_ranges

    sigmas = [float(x) for x in args.sigma.split(",")]
    lines = [
        "# Well-tempered metadynamics on the three CVs of",
        "# bioRxiv 10.1101/2025.11.28.690881 (their method was meta-eABF; PLUMED's DRR",
        "# module was unavailable here, so this is WT-metaD on the same CVs -- the",
        "# deviation must be stated whenever these free energies are compared).",
        "UNITS LENGTH=nm ENERGY=kj/mol TIME=ps",
        *(["RESTART"] if args.restart else []),
        "# Reassemble each PROTEIN CHAIN before any CoM is taken.",
        "# One ENTITY per chain, NOT the whole system: applying WHOLEMOLECULES to all",
        "# ~69k atoms (incl. ~62k waters) cost an 8x slowdown (631 -> 78 ns/day), and",
        "# water needs no reassembly. Each entity must be one connected molecule.",
        "WHOLEMOLECULES " + " ".join(
            f"ENTITY{n}={lo}-{hi}" for n, (lo, hi) in enumerate(protein_chain_ranges)),
        "",
    ]
    for cv in CVS:
        # PLUMED atom serials are 1-based.
        a = ",".join(str(i + 1) for i in groups[cv]["a"])
        c = ",".join(str(i + 1) for i in groups[cv]["c"])
        lines += [f"{cv}_a: COM ATOMS={a}", f"{cv}_c: COM ATOMS={c}",
                  f"{cv}: DISTANCE ATOMS={cv}_a,{cv}_c", ""]

    bias_cvs = [c.strip() for c in args.bias_cvs.split(",")]
    bad = [c for c in bias_cvs if c not in CVS]
    if bad:
        raise SystemExit(f"unknown CV(s) {bad}; choose from {list(CVS)}")
    if len(sigmas) != len(bias_cvs):
        raise SystemExit(f"--sigma has {len(sigmas)} values but --bias-cvs has "
                         f"{len(bias_cvs)}: {bias_cvs}")
    arg = ",".join(bias_cvs)
    all_arg = ",".join(CVS)
    lines += [
        "METAD ...",
        "   LABEL=metad",
        f"   ARG={arg}",
        f"   SIGMA={','.join(str(s) for s in sigmas)}",
        f"   HEIGHT={args.height}",
        f"   BIASFACTOR={args.biasfactor}",
        f"   PACE={args.pace}",
        "   TEMP=310",
        f"   GRID_MIN={','.join(str(RANGES[c][0]/10 - 0.3) for c in bias_cvs)}",
        f"   GRID_MAX={','.join(str(RANGES[c][1]/10 + 0.3) for c in bias_cvs)}",
        "   CALC_RCT",
        # The conda-forge PLUMED is a *nompi* build, so WALKERS_MPI cannot be used.
        # The file-based scheme shares hills through WALKERS_DIR instead.
        *( [f"   WALKERS_N={args.walkers}",
            f"   WALKERS_ID={args.walker_id}",
            f"   WALKERS_DIR={args.walkers_dir}",
            "   WALKERS_RSTRIDE=100"] if args.walkers > 1 else [] ),
        "   FILE=HILLS",
        "... METAD",
        "",
        "# Keep sampling inside the paper's ranges; without walls the bias wanders into",
        "# unphysical territory and wastes the budget.",
    ]
    for cv in bias_cvs:
        lo, hi = RANGES[cv]
        lines += [
            f"lw_{cv}: LOWER_WALLS ARG={cv} AT={lo/10:.3f} KAPPA=2000.0",
            f"uw_{cv}: UPPER_WALLS ARG={cv} AT={hi/10:.3f} KAPPA=2000.0",
        ]
    # Print every CV, not just the biased ones, so S2 remains available for analysis.
    lines += ["", f"PRINT ARG={all_arg},metad.bias STRIDE={args.colvar_stride} FILE=COLVAR", ""]

    out = args.out or (args.run_dir / "plumed.dat")
    out.write_text("\n".join(l for l in lines if l is not None) + "\n")
    (args.run_dir / "plumed_groups.json").write_text(json.dumps(
        {"groups": {k: {t: [i + 1 for i in v] for t, v in g.items()}
                    for k, g in groups.items()}, **report}, indent=2))
    print(f"wrote {out} and {args.run_dir}/plumed_groups.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
