"""Plot the WT-metaD v2 FES (S1, S3): WT Asp143 system, walls S1 17-28 / S3 8-18 A.

Copy of plot_fes_v2basins.py adapted to the metaD v2 run (2026-11-07); --total-ns labels
the simulated time because COLVAR time restarts each phase.
Original header:

Variant of plot_fes.py (2026-11-07): the old "B" marker was the mean of a B run that
collapsed during a flawed prep; it is shown greyed as superseded. The v2 wild-type
basins (9 ns staged equilibration + 5 ns unbiased, last 4 ns, distorted pair) are drawn
with +/-1 sigma boxes, and each box reports F_min, the Boltzmann-integrated basin free
energy and the fraction of its cells the walker actually visited.

Usage: python src/plot_fes.py FES.dat COLVAR OUT.png [--title ...]
The FES is -bias*(gamma/(gamma-1)) from sum_hills; outside the region the walker
actually visited it is meaningless, so those cells are masked.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("fes"); ap.add_argument("colvar"); ap.add_argument("out")
ap.add_argument("--title", default="TNF apo trimer (WT): WT-metaD v2 free energy surface")
ap.add_argument("--total-ns", type=float, default=None)
ap.add_argument("--fmax", type=float, default=12.0, help="kcal/mol colour cap")
a = ap.parse_args()

d = np.loadtxt(a.fes, comments="#")
s1u, s3u = np.unique(d[:, 0]), np.unique(d[:, 1])
F = d[:, 2].reshape(len(s3u), len(s1u)) / 4.184          # kJ -> kcal/mol, rows = S3
cv = np.loadtxt(a.colvar, comments="#")
t_ns = (cv[:, 0] - cv[0, 0]) / 1000

# mask cells farther than ~2 bins from any visited COLVAR point
H, _, _ = np.histogram2d(cv[:, 3], cv[:, 1], bins=[np.append(s3u, s3u[-1] + (s3u[1] - s3u[0])),
                                                 np.append(s1u, s1u[-1] + (s1u[1] - s1u[0]))])
from scipy.ndimage import binary_dilation
visited = binary_dilation(H > 0, iterations=2)
F = np.where(visited, F, np.nan)
F -= np.nanmin(F)

ramp = ["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
        "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]
cmap = LinearSegmentedColormap.from_list("blue_seq", ramp)
INK, INK2, SURF = "#0b0b0b", "#52514e", "#fcfcfb"

fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
X, Y = np.meshgrid(s1u * 10, s3u * 10)          # nm -> A
pc = ax.pcolormesh(X, Y, np.clip(F, 0, a.fmax), cmap=cmap, vmin=0, vmax=a.fmax, shading="auto", rasterized=True)
ax.contour(X, Y, F, levels=np.arange(1, a.fmax, 1), colors="white", linewidths=0.5, alpha=0.55)

# walls (paper's sampling ranges)
for x in (17.0, 28.0):
    ax.axvline(x, color=INK2, lw=1, ls=(0, (4, 3)))
for y in (8.0, 18.0):
    ax.axhline(y, color=INK2, lw=1, ls=(0, (4, 3)))
ax.text(27.9, 18.12, "upper wall S3 = 18 Å", ha="right", va="bottom", fontsize=8, color=INK2)
ax.text(27.9, 7.88, "lower wall S3 = 8 Å", ha="right", va="top", fontsize=8, color=INK2)

# v2 basins (mean, sigma) and reference points
BASINS = {"A (v2 WT)": (20.49, 9.46, 0.35, 0.23), "B (v2 WT)": (23.95, 12.67, 0.65, 0.58)}
from matplotlib.patches import Rectangle
for lab, (x, y, sx, sy) in BASINS.items():
    ax.add_patch(Rectangle((x - sx, y - sy), 2 * sx, 2 * sy, fill=False, ec=INK, lw=1.4, zorder=5))
pts = [
    (20.49, 9.46, "o", "A: v2 WT basin (\u00b11\u03c3 box)", (10, -14), INK),
    (23.95, 12.67, "s", "B: v2 WT basin (\u00b11\u03c3 box)", (12, 10), INK),
    (19.9, 9.3, "^", "1TNF crystal", (-8, 8), INK),
    (24.2, 14.0, "D", "7JRA crystal", (-8, 8), INK),
]
for x, y, m, lab, off, col in pts:
    ax.plot(x, y, m, ms=9, mfc="white" if col == INK else "#e6e5df", mec=col, mew=1.6, zorder=6)
    ax.annotate(lab, (x, y), xytext=off, textcoords="offset points", fontsize=8.5, color=col,
                ha="left" if off[0] > 0 else "right", va="center", zorder=7,
                bbox=dict(boxstyle="round,pad=0.2", fc=SURF, ec="none", alpha=0.85))

ax.set_xlim(16.5, 28.5); ax.set_ylim(7.0, 19.0)
ax.set_xlabel("S1  (Å, CoM distance, subunit pair)", color=INK)
ax.set_ylabel("S3  (Å, CoM distance, subunit pair)", color=INK)
for sp in ax.spines.values():
    sp.set_color("#c3c2b7")
ax.tick_params(colors=INK2)
cb = fig.colorbar(pc, ax=ax, pad=0.02, extend="max")
cb.set_label("Free energy (kcal/mol)", color=INK)
cb.outline.set_visible(False)
ax.set_title(a.title, color=INK, fontsize=11, loc="left", pad=18)
ax.text(0, 1.012, f"{(a.total_ns or t_ns[-1]):.0f} ns, 2 CVs biased (S1, S3); NOT CONVERGED. Blank = never visited; "
        "values far from visited region unreliable", transform=ax.transAxes, fontsize=7.5, color=INK2, va="bottom")
fig.tight_layout()
fig.savefig(a.out, facecolor=SURF)
print("wrote", a.out)
# basin values
kT = 0.0019872 * 310
Fall = d[:, 2].reshape(len(s3u), len(s1u)) / 4.184
Fall -= np.nanmin(np.where(visited, Fall, np.nan))
def box(x, y, sx, sy):
    sel = (np.abs(X - x) <= sx) & (np.abs(Y - y) <= sy)
    vis = sel & visited
    Fv = Fall[vis]
    G = -kT * np.log(np.exp(-Fv / kT).sum()) if vis.any() else np.nan
    return sel.sum(), vis.sum(), (np.min(Fv) if vis.any() else np.nan), G
res = {}
for lab, (x, y, sx, sy) in BASINS.items():
    n, nv, fmin, G = box(x, y, sx, sy); res[lab] = G
    print(f"{lab:10s} box cells {n:4d}, visited {nv:4d} ({100*nv/max(n,1):.0f}%), F_min {fmin:.2f}, basin G {G:.2f} kcal/mol")
print("dG(B - A), Boltzmann-integrated over the +/-1 sigma boxes: %.2f kcal/mol" % (res["B (v2 WT)"] - res["A (v2 WT)"]))
def at(x, y, r=0.5):
    sel = (np.hypot(X - x, Y - y) < r) & np.isfinite(F)
    return np.nanmin(F[sel]) if sel.any() else np.nan
for x, y, m, lab, _, _ in pts:
    print(f"{lab:40s} F_min within 0.5 A = {at(x, y):.2f} kcal/mol")
print("global minimum at S1 %.2f S3 %.2f A" % (X.flat[np.nanargmin(F)], Y.flat[np.nanargmin(F)]))
# time spent in each box
for lab, (x, y, sx, sy) in BASINS.items():
    inb = (np.abs(cv[:, 1] * 10 - x) <= sx) & (np.abs(cv[:, 3] * 10 - y) <= sy)
    print(f"{lab:10s} COLVAR frames inside box: {inb.sum()} of {len(cv)} ({100*inb.mean():.1f}%)")
