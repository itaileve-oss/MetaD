export MAMBA_ROOT_PREFIX=/opt/micromamba
eval "$(/opt/micromamba/bin/micromamba shell hook -s bash)"
micromamba activate tnfmd
micromamba install -y -c conda-forge plumed openmm-plumed py-plumed
echo "=== plumed version ==="
plumed info --version
echo "=== modules (looking for drr = eABF) ==="
plumed config show 2>/dev/null | head -40 || true
echo "=== does DRR exist? ==="
plumed --no-mpi manual --action DRR 2>&1 | head -5
echo "=== EXTENDED_LAGRANGIAN? ==="
plumed --no-mpi manual --action EXTENDED_LAGRANGIAN 2>&1 | head -5
echo "=== openmmplumed import ==="
python -c "from openmmplumed import PlumedForce; print(\"openmm-plumed OK\")"
