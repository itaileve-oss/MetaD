#!/usr/bin/env bash
# Clean WILD-TYPE (Asp143) systems for the v2 metaD, on the metaD instance only.
# Same prep_system.py + equil_protocol.py (~9 ns staged restrained equilibration) and
# bioRxiv settings as the Gen-COMPAS endpoints, but built here independently.
set -euo pipefail
cd /workspace/tnf
PY=/opt/micromamba/envs/tnfmd/bin/python
mkdir -p systems_wt
$PY src/prep_system.py data/1TNF.pdb systems_wt/A --mutate LEU-143-ASP --renumber-start 6 --fix-only
cp systems_wt/A/fixed.pdb systems_wt/A_fixed_ref.pdb
COMMON="--renumber-start 6 --platform CUDA"
($PY -u src/prep_system.py data/1TNF.pdb systems_wt/A --mutate LEU-143-ASP $COMMON > systems_wt/A.prep.log 2>&1) &
($PY -u src/prep_system.py data/7JRA.pdb systems_wt/B --chain-map C:A,B:B,A:C --res-range 82-233 \
    --graft-loops-from C --variants-from systems_wt/A_fixed_ref.pdb --pocket-ref data/7JRA.pdb \
    $COMMON > systems_wt/B.prep.log 2>&1) &
wait
echo "=== WT systems done ==="
