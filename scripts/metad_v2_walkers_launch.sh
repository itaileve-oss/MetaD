#!/usr/bin/env bash
set -euo pipefail
export MAMBA_ROOT_PREFIX=/opt/micromamba
eval "$(/opt/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate tnfmd
V=/workspace/tnf/metad_v2; NS=${NS:-380}
cd $V/w0; python ../../src/run_metad.py . --ns $NS --plumed plumed.dat --state-pdb equilibrated_A.pdb --state-xml equilibrated_A.xml --out-prefix mwA --save-ps 100 --checkpoint-ns 10 --resume > run.log 2>&1 &
sleep 60
cd $V/w1; python ../../src/run_metad.py . --ns $NS --plumed plumed.dat --state-pdb equilibrated_B.pdb --state-xml equilibrated_B.xml --out-prefix mwB --save-ps 100 --checkpoint-ns 10 > run.log 2>&1 &
wait
echo "=== walkers done $(date -u +%FT%TZ) ==="
