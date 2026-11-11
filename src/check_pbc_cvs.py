"""Independent recomputation of S1/S3 from saved metaD frames, to verify PLUMED's PBC handling.

For each frame computes:
  naive : COM of raw (as-written) coordinates, plain Euclidean distance (no PBC at all)
  pbc   : each chain made whole along bonds, COM, then exact minimum-image distance
          (search over all -1..1 lattice shifts, valid for triclinic boxes)
and compares with PLUMED's COLVAR at the matching time (offset found by search).
Masses: HMR-adjusted (as the MD engine sees them) and plain element masses.
Usage: python check_pbc_cvs.py TOP.pdb TRAJ.dcd COLVAR PLUMED.dat [--hmr 1.5] [--last N]
"""
import argparse, itertools, re
import numpy as np
import mdtraj as md

ap = argparse.ArgumentParser()
ap.add_argument("top"); ap.add_argument("dcd"); ap.add_argument("colvar"); ap.add_argument("plumed")
ap.add_argument("--hmr", type=float, default=None)
ap.add_argument("--last", type=int, default=300)
ap.add_argument("--save-ps", type=float, default=100.0)
a = ap.parse_args()

txt = open(a.plumed).read()
groups = {k: [int(i) - 1 for i in v.split(",")]
          for k, v in re.findall(r"^(S[13]_[ac]): COM ATOMS=([\d,]+)", txt, re.M)}
assert len(groups) == 4, groups

top = md.load_topology(a.top)
for g, idx in groups.items():
    print(g, [f"{top.atom(i).residue.chain.index}:{top.atom(i).residue}:{top.atom(i).name}" for i in idx])

t = md.load(a.dcd, top=top)
t = t[-a.last:] if a.last and t.n_frames > a.last else t
first_frame = md.load(a.dcd, top=top).n_frames - t.n_frames
print("frames analysed:", t.n_frames, "starting at frame index", first_frame)
print("box (nm) first frame lengths", t.unitcell_lengths[0], "angles", t.unitcell_angles[0])

# masses
m_plain = np.array([at.element.mass for at in top.atoms])
m_hmr = m_plain.copy()
if a.hmr:
    for b1, b2 in top.bonds:
        for h, x in ((b1, b2), (b2, b1)):
            if h.element.symbol == "H" and x.element.symbol != "H":
                m_hmr[x.index] -= a.hmr - m_plain[h.index]
                m_hmr[h.index] = a.hmr

raw = t.xyz.copy()
whole = t.make_molecules_whole(inplace=False).xyz
box = t.unitcell_vectors  # (F,3,3) rows = a,b,c
shifts = np.array(list(itertools.product((-1, 0, 1), repeat=3)))

def com(x, idx, m):
    w = m[idx] / m[idx].sum()
    return (x[:, idx, :] * w[None, :, None]).sum(1)

def mic(d):
    best = np.full(len(d), np.inf)
    for s in shifts:
        v = d + np.einsum("j,fjk->fk", s, box)
        best = np.minimum(best, np.linalg.norm(v, axis=1))
    return best

res = {}
for cv in ("S1", "S3"):
    ga, gc = groups[cv + "_a"], groups[cv + "_c"]
    for mname, m in (("hmr", m_hmr), ("plain", m_plain)):
        res[(cv, "naive", mname)] = np.linalg.norm(com(raw, ga, m) - com(raw, gc, m), axis=1)
        res[(cv, "pbc", mname)] = mic(com(whole, ga, m) - com(whole, gc, m))
    # intra-group spread: if a group straddles the box edge, raw spread is huge
    for g in (ga, gc):
        sp = np.linalg.norm(raw[:, g, :] - raw[:, g, :].mean(1, keepdims=True), axis=2).max()
        print(f"{cv} group max atom-to-centroid distance in RAW coords: {sp:.2f} nm")

cvd = np.loadtxt(a.colvar, comments="#")
ct, cS1, cS3 = cvd[:, 0], cvd[:, 1], cvd[:, 3]
# frame k of the dcd is written at step interval*(k+1); find best time offset to COLVAR
ref = res[("S1", "pbc", "hmr")]
best = None
for off in np.arange(-5000, 5001, 10.0):
    tf = ct[0] - 1000 + off + (np.arange(first_frame, first_frame + t.n_frames) + 1) * a.save_ps
    j = np.searchsorted(ct, tf)
    ok = (j < len(ct)) & (np.abs(ct[np.clip(j, 0, len(ct) - 1)] - tf) < 1e-3)
    if ok.sum() < 0.8 * len(tf):
        continue
    err = np.abs(cS1[j[ok]] - ref[ok]).mean()
    if best is None or err < best[0]:
        best = (err, off, j, ok)
err, off, j, ok = best
print(f"best COLVAR time offset {off:+.0f} ps, matched frames {ok.sum()}/{len(ok)}")
for cv, col in (("S1", cS1), ("S3", cS3)):
    for kind in ("pbc", "naive"):
        for mname in ("hmr", "plain"):
            d = res[(cv, kind, mname)][ok] - col[j[ok]]
            print(f"{cv} {kind:5s} {mname:5s}: mean|diff| {np.abs(d).mean()*10:.4f} A, max|diff| {np.abs(d).max()*10:.4f} A")
    print(f"{cv} naive-vs-pbc max difference over all frames: "
          f"{np.abs(res[(cv,'naive','hmr')] - res[(cv,'pbc','hmr')]).max()*10:.3f} A")
print("S1 range (pbc) %.2f-%.2f A, S3 range %.2f-%.2f A" % (
    res[("S1","pbc","hmr")].min()*10, res[("S1","pbc","hmr")].max()*10,
    res[("S3","pbc","hmr")].min()*10, res[("S3","pbc","hmr")].max()*10))
