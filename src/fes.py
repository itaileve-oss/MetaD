"""Build the free energy surface from whatever hills exist, and judge convergence.

Safe to run at any time while the metadynamics is still going: HILLS is appended
continuously, so this gives the FES earned so far. That is what makes "set a long run
and stop when you like" a sound strategy.

Convergence is judged by rebuilding the FES from increasing fractions of the hills and
comparing: if the last few estimates agree, the surface has stopped moving. A single
FES on its own says nothing about convergence.

Usage:
    python src/fes.py HILLS_FILE [--plumed /opt/plumed292/bin/plumed] [--blocks 4]
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np

KJ_PER_KCAL = 4.184


def sum_hills(plumed: str, hills: Path, outdir: Path, stride: int | None = None,
              mintozero: bool = True) -> list[Path]:
    cmd = [plumed, "--no-mpi", "sum_hills", "--hills", str(hills),
           "--outfile", str(outdir / "fes_")]
    if mintozero:
        cmd.append("--mintozero")
    if stride:
        cmd += ["--stride", str(stride)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
    if res.returncode != 0:
        raise SystemExit(f"sum_hills failed:\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
    return sorted(outdir.glob("fes_*.dat"))


def read_fes(path: Path):
    """Return (coords array Nx2 or Nx1, free energy in kJ/mol)."""
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            rows.append([float(x) for x in parts])
    if not rows:
        return None, None
    arr = np.array(rows)
    ncv = 2 if arr.shape[1] >= 4 else 1
    return arr[:, :ncv], arr[:, ncv]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("hills", type=Path)
    ap.add_argument("--plumed", default="/opt/plumed292/bin/plumed")
    ap.add_argument("--blocks", type=int, default=4,
                    help="rebuild the FES from this many cumulative fractions")
    ap.add_argument("--out", type=Path, default=None)
    # Basin locations from the unbiased runs (nm): state A (1TNF) and state B (7JRA,
    # ligand removed). dG between THESE is what compares to the paper's 5-7 kcal/mol;
    # the FES span across the whole grid does not.
    ap.add_argument("--basin-a", default="2.061,0.897", help="S1,S3 in nm")
    ap.add_argument("--basin-b", default="2.265,0.981", help="S1,S3 in nm")
    ap.add_argument("--basin-radius", type=float, default=0.05,
                    help="nm; minimum of the FES within this radius is taken")
    args = ap.parse_args()

    n_hills = sum(1 for l in args.hills.read_text().splitlines()
                  if l.strip() and not l.startswith("#"))
    print(f"{args.hills}: {n_hills} hills")
    if n_hills < 50:
        print("too few hills to be worth summing yet")
        return 0

    outdir = args.out or args.hills.parent / "fes"
    outdir.mkdir(parents=True, exist_ok=True)

    # Cumulative estimates. The stride must divide the hill count closely, otherwise the
    # last file adds only a handful of hills and its RMS change is spuriously tiny --
    # which looks like convergence and is not.
    stride = max(1, n_hills // args.blocks)
    if n_hills % stride and n_hills % stride < stride // 2:
        stride = max(1, n_hills // max(1, args.blocks - 1))
    files = sum_hills(args.plumed, args.hills, outdir, stride=stride)
    print(f"built {len(files)} cumulative FES estimates (stride {stride} hills)")

    prev = None
    for i, f in enumerate(files):
        coords, fe = read_fes(f)
        if fe is None:
            continue
        rng = fe.max() - fe.min()
        line = (f"  {f.name}: {len(fe)} grid points, span "
                f"{rng:7.2f} kJ/mol = {rng / KJ_PER_KCAL:6.2f} kcal/mol")
        if prev is not None and len(prev) == len(fe):
            # Drift between successive cumulative estimates: the convergence signal.
            d = fe - prev
            line += (f" | RMS change vs previous {np.sqrt((d ** 2).mean()):.2f} kJ/mol"
                     f" ({np.sqrt((d ** 2).mean()) / KJ_PER_KCAL:.2f} kcal/mol)")
        print(line)
        prev = fe

    final = files[-1]
    coords, fe = read_fes(final)
    print(f"\nfinal estimate: {final}")

    def basin_min(target):
        t = np.array([float(x) for x in target.split(",")])
        if coords.shape[1] != len(t):
            return None, 0
        d = np.linalg.norm(coords - t, axis=1)
        sel = d <= args.basin_radius
        if not sel.any():
            k = int(np.argmin(d))
            return float(fe[k]), 1
        return float(fe[sel].min()), int(sel.sum())

    fa, na = basin_min(args.basin_a)
    fb, nb = basin_min(args.basin_b)
    if fa is not None and fb is not None:
        dg_kj = fb - fa
        print(f"  basin A ({args.basin_a}) min {fa:8.2f} kJ/mol  [{na} grid pts]")
        print(f"  basin B ({args.basin_b}) min {fb:8.2f} kJ/mol  [{nb} grid pts]")
        print(f"  ** dG(B - A) = {dg_kj:+.2f} kJ/mol = {dg_kj / KJ_PER_KCAL:+.2f} kcal/mol **")
        print("     compare published sTNF: asymmetric ~5-7 kcal/mol ABOVE symmetric.")
        print("     CAVEATS: WT-metaD here vs their meta-eABF; less sampling; 1.0 nm")
        print("     cutoff vs their 1.2 nm; 2 CVs biased vs their 3.")
    print(f"  (FES span over the whole grid is {(fe.max() - fe.min()) / KJ_PER_KCAL:.2f} "
          "kcal/mol -- NOT the basin dG, it includes wall/edge regions)")
    print("  Judge convergence from the RMS-change column, NOT from one surface.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
