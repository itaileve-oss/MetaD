"""Compute the paper's CVs (and our geometric theta-spread) along a trajectory.

Purpose, in order of importance:
  1. Give the CV fluctuation widths in each basin -> METAD's SIGMA, the one bias
     parameter bioRxiv 10.1101/2025.11.28.690881 did not report.
  2. Metastability check on state B: does the distortion persist without the ligand?
  3. Signal-to-noise on our own theta-spread CV, which so far was measured only on
     crystal structures.

Group resolution is imported from make_plumed so there is ONE implementation of the
numbering logic (paper -> deposited -> pdbfixer-renumbered).

Usage:
    python src/analyze_cvs.py RUN_DIR --traj unbiased.dcd \
        --orig data/1TNF.pdb --ecd-offset 76 --pair A,B
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import mdtraj as md
import numpy as np

from make_plumed import CVS, ca_index_map, first_resnum_per_chain
import openmm.app as app


def resolve(run_dir: Path, orig: Path, ecd_offset: int, pair: str):
    chain_a, chain_c = [c.strip() for c in pair.split(",")]
    pdb = app.PDBFile(str(run_dir / "equilibrated.pdb"))
    idx = ca_index_map(pdb.topology)
    offsets = {ch: n - 1 for ch, n in first_resnum_per_chain(orig).items()}
    out = {}
    for cv, (ga, gc) in CVS.items():
        side = {}
        for tag, chain, grouplist in (("a", chain_a, ga), ("c", chain_c, gc)):
            got = []
            for grp in grouplist:
                for paper_res in grp:
                    key = (chain, paper_res - ecd_offset - offsets.get(chain, 0))
                    if key in idx:
                        got.append(idx[key])
            side[tag] = got
        out[cv] = side
    return out, idx, offsets


def theta_spread(traj: md.Trajectory, idx, chains=("A", "B", "C")):
    """Std of the three inter-subunit rotation angles, per frame (our own CV)."""
    per_chain = {c: sorted(r for (ch, r) in idx if ch == c) for c in chains}
    common = sorted(set.intersection(*[set(per_chain[c]) for c in chains]))
    sel = {c: np.array([idx[(c, r)] for r in common]) for c in chains}
    out = []
    for frame in traj.xyz:
        angs = []
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            P, Q = frame[sel[chains[a]]], frame[sel[chains[b]]]
            pc, qc = P.mean(0), Q.mean(0)
            u, _, vt = np.linalg.svd((P - pc).T @ (Q - qc))
            d = np.sign(np.linalg.det(vt.T @ u.T))
            R = vt.T @ np.diag([1, 1, d]) @ u.T
            angs.append(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
        out.append(np.std(angs))
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--traj", default="unbiased.dcd")
    ap.add_argument("--orig", type=Path, required=True)
    ap.add_argument("--ecd-offset", type=int, required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--skip-ns", type=float, default=1.0, help="discard as re-equilibration")
    ap.add_argument("--frame-ps", type=float, default=10.0)
    args = ap.parse_args()

    d = args.run_dir
    groups, idx, offsets = resolve(d, args.orig, args.ecd_offset, args.pair)
    traj = md.load(str(d / args.traj), top=str(d / "equilibrated.pdb"))
    # The trimer can straddle the periodic boundary, putting two subunit COMs in
    # different images and yielding ~box-length "distances" (~80 A here). State A
    # happened to be whole, state B was not. PLUMED handles this with WHOLEMOLECULES;
    # the analysis must do the equivalent.
    protein = traj.topology.select("protein")
    try:
        traj.image_molecules(inplace=True,
                             anchor_molecules=[set(traj.topology.atom(i)
                                                   for i in protein)])
        print("  periodic images fixed with image_molecules()")
    except Exception as exc:  # noqa: BLE001
        print(f"  WARNING image_molecules failed ({type(exc).__name__}: {exc}); "
              "falling back to manual unwrap of subunit COMs")
    skip = int(args.skip_ns * 1000 / args.frame_ps)
    traj = traj[skip:] if len(traj) > skip else traj
    xyz = traj.xyz * 10.0  # nm -> Angstrom

    print(f"{d.name}: {len(traj)} frames analysed (first {args.skip_ns} ns discarded)")
    report = {"run": d.name, "n_frames": len(traj), "pair": args.pair}
    sigmas_nm = []
    for cv, side in groups.items():
        a = np.array(side["a"]); c = np.array(side["c"])
        series = np.linalg.norm(xyz[:, a].mean(1) - xyz[:, c].mean(1), axis=1)
        mean, std = float(series.mean()), float(series.std())
        if mean > 50.0:
            raise SystemExit(
                f"ABORT: {cv} mean {mean:.1f} A is implausible for a CV whose reference "
                "values are 8-25 A. This is almost certainly a periodic-image problem "
                "(value near the box length) or a residue-mapping error."
            )
        report[cv] = {"mean_A": round(mean, 2), "std_A": round(std, 3),
                      "min_A": round(float(series.min()), 2),
                      "max_A": round(float(series.max()), 2)}
        # Common practice: SIGMA ~ the CV's fluctuation in the basin (here std, in nm).
        sigmas_nm.append(round(std / 10.0, 4))
        print(f"  {cv}: mean {mean:6.2f} A   std {std:5.3f} A   "
              f"range [{series.min():.2f}, {series.max():.2f}]")

    th = theta_spread(traj, idx)
    report["theta_spread_deg"] = {"mean": round(float(th.mean()), 2),
                                  "std": round(float(th.std()), 2),
                                  "min": round(float(th.min()), 2),
                                  "max": round(float(th.max()), 2)}
    print(f"  theta-spread (our CV): mean {th.mean():.2f} deg  std {th.std():.2f}  "
          f"range [{th.min():.2f}, {th.max():.2f}]")
    print(f"  -> suggested METAD SIGMA (nm): {','.join(str(s) for s in sigmas_nm)}")
    report["suggested_sigma_nm"] = sigmas_nm

    (d / "cv_analysis.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote {d}/cv_analysis.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
