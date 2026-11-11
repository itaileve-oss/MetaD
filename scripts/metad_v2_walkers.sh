#!/usr/bin/env bash
# metaD v2 -> 2 CONCURRENT walkers (A + B) sharing one bias (user OK 2026-11-08 ~8:30 PM PST),
# replacing the A/B phase alternation, which produced a seesaw in dG(B-A).
# Walker 0 = A, continues phase 3 from metadA_p3b.chk (140 ns); walker 1 = B, fresh from equilibrated_B.
# All previous hills (p1+p2+p3 to 140 ns) become walkers_shared/HILLS.0. Same CVs/sigma/height/BF/pace.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/opt/micromamba
eval "$(/opt/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate tnfmd
T=/workspace/tnf; V=$T/metad_v2; R=$V/run; NS=${NS:-380}
pkill -f "bash metad_v2_resume.sh" || true; pkill -f "out-prefix metadA_p3b" || true; sleep 15
pgrep -f run_metad.py && { echo "run_metad still alive"; exit 1; }
B=$V/backup_before_walkers; mkdir -p $B; cp $R/HILLS $R/COLVAR $R/metadA_p3b.chk $R/metadA_p3b.dcd $R/metadA_p3b.log $B/
cd $R
python - <<PY
for fn in ("HILLS","COLVAR"):
    L=open(fn).read().splitlines(keepends=True); n=len(L)
    while L and (L[-1].startswith("#") or float(L[-1].split()[0])>140000.001): L.pop()
    open(fn,"w").writelines(L); print(fn,"removed",n-len(L),"lines; last t",L[-1].split()[0])
PY
mv COLVAR COLVAR.p3_A
mkdir -p $V/walkers_shared $V/w0 $V/w1
cp HILLS $V/walkers_shared/HILLS.0
for P in A B; do cp equilibrated_$P.pdb equilibrated_$P.xml $V/w0/; cp equilibrated_$P.pdb equilibrated_$P.xml $V/w1/; done
cp metadA_p3b.chk $V/w0/mwA.chk
cd $T
EXP_A=$(python -c "import json;f=json.load(open('systems_wt/A/prep_report.json'))['equilibration']['final'];print(f\"{f['S1']:.2f},{f['S2']:.2f},{f['S3']:.2f}\")")
EXP_B=$(python -c "import json;f=json.load(open('systems_wt/B/prep_report.json'))['equilibration']['final'];print(f\"{f['S1']:.2f},{f['S2']:.2f},{f['S3']:.2f}\")")
COMMONP="--orig $T/systems_wt/A/fixed.pdb --ecd-offset 76 --pair C,A --sigma 0.025,0.015 --tol 0.5 --walkers 2 --walkers-dir ../walkers_shared --restart"
python src/make_plumed.py $V/w0 $COMMONP --walker-id 0 --state-pdb equilibrated_A.pdb --expect $EXP_A --out $V/w0/plumed.dat
python src/make_plumed.py $V/w1 $COMMONP --walker-id 1 --state-pdb equilibrated_B.pdb --expect $EXP_B --out $V/w1/plumed.dat
# everything except the walker lines must equal the plumed used so far
for w in w0 w1; do
  diff <(grep -vE "WALKERS_" $V/$w/plumed.dat) <(cat $R/plumed_A_restart.dat) && echo "$w plumed == previous (except walkers)" || { echo "ABORT: $w plumed differs"; exit 1; }
done
cd $V/w0; python ../../src/run_metad.py . --ns $NS --plumed plumed.dat --state-pdb equilibrated_A.pdb --state-xml equilibrated_A.xml \
  --out-prefix mwA --save-ps 100 --checkpoint-ns 10 --resume > run.log 2>&1 &
sleep 60
cd $V/w1; python ../../src/run_metad.py . --ns $NS --plumed plumed.dat --state-pdb equilibrated_B.pdb --state-xml equilibrated_B.xml \
  --out-prefix mwB --save-ps 100 --checkpoint-ns 10 > run.log 2>&1 &
wait
echo "=== walkers done $(date -u +%FT%TZ) ==="
