"""Salt-bridge switch CV on metaD trajectories (PBC-safe: pairwise minimum-image distances).

Glu104 of chain X pairs with Arg103 of chain X-1 in state A, and with Lys112 of chain X+1
in state B (from the Gen-COMPAS cross-pull analysis). Per frame, for each of the 3 cyclic
interfaces, min N-O distance for:
  R103: Arg103(NE,NH1,NH2) of chain i  -- Glu104(OE1,OE2) of chain i+1
  K112: Lys112(NZ) of chain i+1        -- Glu104(OE1,OE2) of chain i
Switching contacts c(d) = (1-(d/r0)^6)/(1-(d/r0)^12), r0 = 0.4 nm.
s_switch = sum_i [c(K112_i) - c(R103_i)] ranges -3 (all A-like) .. +3 (all B-like).
Also reports A-only intra bridges Arg6-Glu127 and Arg32-Glu146.
Usage: python saltbridge_cv.py OUT.tsv label:top.pdb:traj.dcd:colvar [...]
"""
import sys
import numpy as np
import mdtraj as md

R0 = 0.4
OFF = 5  # topology is renumbered 1-152; TNF ECD numbering = topology + 5 (resid 1 = Arg6)


def sw(d):
    x = d / R0
    return (1 - x**6) / (1 - x**12 + 1e-12)


def atoms(top, chain, resSeq, names):
    idx = [a.index for a in top.chain(chain).atoms
            if a.residue.resSeq == resSeq - OFF and a.name in names]
    assert idx, (chain, resSeq, names)
    return idx


def mindist(t, g1, g2):
    pairs = np.array([(i, j) for i in g1 for j in g2])
    return md.compute_distances(t, pairs, periodic=True).min(1)


ARG, LYS, GLU = ("NE", "NH1", "NH2"), ("NZ",), ("OE1", "OE2")
out = open(sys.argv[1], "w")
out.write("label\tframe\tS1\tS3\ts_switch\t" + "\t".join(
    [f"R103_{i}" for i in "ABC"] + [f"K112_{i}" for i in "ABC"] +
    [f"R6E127_{i}" for i in "ABC"] + [f"R32E146_{i}" for i in "ABC"]) + "\n")
for spec in sys.argv[2:]:
    label, top_f, dcd, colvar = spec.split(":")
    t = md.load(dcd, top=top_f)
    top = t.topology
    assert top.n_chains >= 3
    cols = {}
    for i in range(3):
        j, k = (i + 1) % 3, (i - 1) % 3
        # interface i: chain i Arg103 -> chain j Glu104 ; chain j Lys112 -> chain i Glu104
        cols[f"R103_{'ABC'[i]}"] = mindist(t, atoms(top, i, 103, ARG), atoms(top, j, 104, GLU))
        cols[f"K112_{'ABC'[i]}"] = mindist(t, atoms(top, j, 112, LYS), atoms(top, i, 104, GLU))
        cols[f"R6E127_{'ABC'[i]}"] = min(
            [mindist(t, atoms(top, i, 6, ARG), atoms(top, x, 127, GLU)) for x in range(3)],
            key=lambda v: v.mean())
        cols[f"R32E146_{'ABC'[i]}"] = mindist(t, atoms(top, i, 32, ARG), atoms(top, i, 146, GLU))
    s = sum(sw(cols[f"K112_{c}"]) - sw(cols[f"R103_{c}"]) for c in "ABC")
    cv = np.loadtxt(colvar, comments="#")
    # dcd frame k <-> COLVAR time t0 + 100 ps * (k+1) (verified offset, see check_pbc_cvs.py)
    tf = cv[0, 0] - 1000 + 100.0 * (np.arange(t.n_frames) + 1)
    idx = np.clip(np.searchsorted(cv[:, 0], tf - 1e-3), 0, len(cv) - 1)
    S1, S3 = cv[idx, 1] * 10, cv[idx, 3] * 10
    names = list(cols)
    for f in range(t.n_frames):
        out.write(f"{label}\t{f}\t{S1[f]:.2f}\t{S3[f]:.2f}\t{s[f]:.3f}\t" +
                  "\t".join(f"{cols[n][f]*10:.2f}" for n in names) + "\n")
    print(f"{label}: {t.n_frames} frames, s_switch mean {s.mean():+.2f} "
          f"(first 10%: {s[:max(1,len(s)//10)].mean():+.2f}, last 10%: {s[-max(1,len(s)//10):].mean():+.2f})",
          flush=True)
out.close()
