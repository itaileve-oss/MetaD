"""Build PLUMED input for metaD v3: CVs S1 + s (salt-bridge switch), S3 printed + walls only.
s = sum_i [c(Lys112(j) NZ - Glu104(i) OE) - c(Arg103(i) N - Glu104(j) OE)], c=1/(1+(d/0.4)^6),
d = smooth minimum distance (PLUMED MIN={BETA}), matching metad_v2/saltbridge_cv.py.
Usage: python make_plumed_s.py top.pdb template_plumed.dat OUT.dat [walker_id] [--driver]"""
import sys, re, mdtraj as md
top_f, tmpl, out = sys.argv[1:4]
wid = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "--driver" else "0"
driver = "--driver" in sys.argv
OFF = 5
top = md.load(top_f).topology
def ser(chain, res, names):
    idx = [a.index + 1 for a in top.chain(chain).atoms if a.residue.resSeq == res - OFF and a.name in names]
    assert idx, (chain, res, names); return ",".join(map(str, idx))
ARG, LYS, GLU = ("NE","NH1","NH2"), ("NZ",), ("OE1","OE2")
t = open(tmpl).read()
head = t.split("METAD ...")[0]
head = head.replace("RESTART\n", "")  # fresh run (and driver)
L = [head.rstrip(), "", "# --- salt-bridge switch CV s (metaD v3) ---"]
terms = []
for i in range(3):
    j = (i + 1) % 3; c = "ABC"[i]
    L.append(f"dR{c}: DISTANCES GROUPA={ser(i,103,ARG)} GROUPB={ser(j,104,GLU)} MIN={{BETA=50}}")
    L.append(f"dK{c}: DISTANCES GROUPA={ser(j,112,LYS)} GROUPB={ser(i,104,GLU)} MIN={{BETA=50}}")
    L.append(f"cR{c}: CUSTOM ARG=dR{c}.min FUNC=1/(1+(x/0.4)^6) PERIODIC=NO")
    L.append(f"cK{c}: CUSTOM ARG=dK{c}.min FUNC=1/(1+(x/0.4)^6) PERIODIC=NO")
    terms += [f"cK{c}", f"cR{c}"]
L.append("s: COMBINE ARG=" + ",".join(terms) + " COEFFICIENTS=" + ",".join(["1","-1"]*3) + " PERIODIC=NO")
if driver:
    L.append("PRINT ARG=S1,S3,s," + ",".join(terms) + " STRIDE=1 FILE=COLVAR_driver")
else:
    L += ["", "METAD ...", "   LABEL=metad", "   ARG=S1,s", "   SIGMA=0.025,0.10", "   HEIGHT=1.2",
          "   BIASFACTOR=15.0", "   PACE=1000", "   TEMP=310", "   GRID_MIN=1.4,-3.5", "   GRID_MAX=3.1,3.5",
          "   CALC_RCT", "   WALKERS_N=2", f"   WALKERS_ID={wid}", "   WALKERS_DIR=../walkers_shared",
          "   WALKERS_RSTRIDE=100", "   FILE=HILLS", "... METAD", "",
          "lw_S1: LOWER_WALLS ARG=S1 AT=1.700 KAPPA=2000.0",
          "uw_S1: UPPER_WALLS ARG=S1 AT=2.800 KAPPA=2000.0",
          "lw_S3: LOWER_WALLS ARG=S3 AT=0.800 KAPPA=2000.0",
          "uw_S3: UPPER_WALLS ARG=S3 AT=1.800 KAPPA=2000.0", "",
          "PRINT ARG=S1,S2,S3,s,metad.bias," + ",".join(terms) + " STRIDE=2500 FILE=COLVAR"]
open(out, "w").write("\n".join(L) + "\n")
