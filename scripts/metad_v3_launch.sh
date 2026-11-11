#!/usr/bin/env bash
# metaD v3: 2 walkers, CVs S1 + s (salt-bridge switch), fresh bias. w0 from A, w1 from B.
set -euo pipefail
export MAMBA_ROOT_PREFIX=/opt/micromamba
eval "$(/opt/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate tnfmd
V=/workspace/tnf/metad_v3; NS=${NS:-200}
cd $V/w0; python ../../src/run_metad.py . --ns $NS --plumed plumed.dat --state-pdb equilibrated_A.pdb --state-xml equilibrated_A.xml --out-prefix m3A --save-ps 100 --checkpoint-ns 10 > run.log 2>&1 &
sleep 60
cd $V/w1; python ../../src/run_metad.py . --ns $NS --plumed plumed.dat --state-pdb equilibrated_B.pdb --state-xml equilibrated_B.xml --out-prefix m3B --save-ps 100 --checkpoint-ns 10 > run.log 2>&1 &
wait
echo "=== v3 walkers done $(date -u +%FT%TZ) ==="
