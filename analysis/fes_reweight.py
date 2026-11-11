"""FES from a 2-CV well-tempered (multiple-walker) metaD run, plus final-bias reweighting.

1. Sums the HILLS Gaussians on a grid. PLUMED writes WT hill heights pre-scaled by
   gamma/(gamma-1), so minus the sum is the FES directly (what `plumed sum_hills` does).
2. Reweights every COLVAR frame with w ~ exp(V_final(x)/kT), V = (gamma-1)/gamma * (-F),
   and projects onto any two printed columns (e.g. an unbiased CV).
3. Reports dG(B-A) between two windows around user-given state centres, and how the
   biased-CV dG evolves with simulation time (the honest convergence check: if it keeps
   swinging, there is no converged number).

Assumes COLVAR columns: time, then the two biased CVs at --bias-cols, and the projection
CVs at --proj-cols (0-based, after `time`). Units follow PLUMED (nm, kJ/mol).

Example (TNF metaD v3: biased S1,s; projected S1,S3):
    python analysis/fes_reweight.py --hills HILLS.0 HILLS.1 --colvar COLVAR.w0 COLVAR.w1 \
        --bias-cols 1 4 --proj-cols 1 3 --proj-scale 10 10 --temp 310 --biasfactor 15 \
        --grid 1.6 2.9 131 -3.5 3.5 141 --out fes.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--hills", nargs="+", required=True)
ap.add_argument("--colvar", nargs="+", required=True)
ap.add_argument("--bias-cols", nargs=2, type=int, default=[1, 2], help="COLVAR columns of the 2 biased CVs")
ap.add_argument("--proj-cols", nargs=2, type=int, default=[1, 2], help="COLVAR columns to project onto")
ap.add_argument("--proj-scale", nargs=2, type=float, default=[1, 1], help="unit scale for projection (10 = nm->A)")
ap.add_argument("--temp", type=float, default=300.0)
ap.add_argument("--biasfactor", type=float, required=True)
ap.add_argument("--grid", nargs=6, type=float, required=True, metavar=("X0", "X1", "NX", "Y0", "Y1", "NY"))
ap.add_argument("--state-a", nargs=2, type=float, help="projection-space centre of state A")
ap.add_argument("--state-b", nargs=2, type=float, help="projection-space centre of state B")
ap.add_argument("--window", type=float, default=0.3, help="half-width of the state windows (projection units)")
ap.add_argument("--time-blocks", type=float, default=10000.0, help="time step (ps) for the dG(t) check")
ap.add_argument("--out", default="fes.png")
a = ap.parse_args()

kTkj = 0.0083145 * a.temp
kT = 0.0019872 * a.temp
g = a.biasfactor
H = np.vstack([np.loadtxt(f, comments="#")[:, :6] for f in a.hills])  # time x y sx sy h
x = np.linspace(a.grid[0], a.grid[1], int(a.grid[2]))
y = np.linspace(a.grid[3], a.grid[4], int(a.grid[5]))
Xg, Yg = np.meshgrid(x, y, indexing="ij")


def fes(h_all):
    F = np.zeros_like(Xg)
    for c in range(0, len(h_all), 2000):
        h = h_all[c:c + 2000]
        F -= np.einsum("k,kij->ij", h[:, 5], np.exp(-0.5 * (((Xg[None] - h[:, 1, None, None]) / h[:, 3, None, None]) ** 2
                                                        + ((Yg[None] - h[:, 2, None, None]) / h[:, 4, None, None]) ** 2)))
    return F - F.min()  # kJ/mol


F = fes(H)
C = [np.loadtxt(f, comments="#") for f in a.colvar]
D = np.vstack(C)
bx, by = a.bias_cols
V = (g - 1) / g * (-RegularGridInterpolator((x, y), F, bounds_error=False, fill_value=None)(D[:, [bx, by]]))
V -= V.max()
w = np.exp(V / kTkj)
w /= w.sum()
px, py = D[:, a.proj_cols[0]] * a.proj_scale[0], D[:, a.proj_cols[1]] * a.proj_scale[1]
print(f"frames {len(D)}; reweighting ESS {1 / np.sum(w ** 2):.0f}")

if a.state_a and a.state_b:
    def win(c):
        return (np.abs(px - c[0]) <= a.window) & (np.abs(py - c[1]) <= a.window)
    pa, pb = w[win(a.state_a)].sum(), w[win(a.state_b)].sum()
    if pa > 0 and pb > 0:
        print(f"dG(B-A), +/-{a.window} windows in projection space: {-kT * np.log(pb / pa):+.2f} kcal/mol")
    else:
        print("one state window is empty: the walkers never visited it")

# dG(t) on the biased CVs: split the first biased CV at its midpoint between the two states
ts = np.arange(a.time_blocks, H[:, 0].max() + 1, a.time_blocks)
for t in ts:
    Ft = fes(H[H[:, 0] <= t]) / 4.184
    p = np.exp(-Ft / kT)
    mid = 0.5 * (x[0] + x[-1])
    print(f"  t <= {t / 1000:6.1f} ns per walker: dG(right-left of {mid:.2f}) on biased CV1 = "
          f"{-kT * np.log(p[x > mid].sum() / p[x <= mid].sum()):+.2f} kcal/mol")

fig, ax = plt.subplots(1, 2, figsize=(14, 5.4))
m0 = ax[0].pcolormesh(x, y, (F / 4.184).T, cmap="viridis", vmin=0, vmax=20, shading="auto")
ax[0].contour(x, y, (F / 4.184).T, levels=range(2, 20, 2), colors="k", linewidths=0.4)
ax[0].set_title("biased CVs (sum of hills)")
fig.colorbar(m0, ax=ax[0], label="F (kcal/mol)")
Hh, e0, e1 = np.histogram2d(px, py, bins=120, weights=w)
Frw = np.where(Hh > 0, -kT * np.log(np.where(Hh > 0, Hh, 1)), np.nan)
Frw -= np.nanmin(Frw)
m1 = ax[1].pcolormesh(e0, e1, Frw.T, cmap="viridis", vmin=0, vmax=8)
ax[1].set_title("reweighted projection (final bias)")
fig.colorbar(m1, ax=ax[1], label="F (kcal/mol)")
fig.savefig(a.out, dpi=130, bbox_inches="tight")
print("wrote", a.out)
