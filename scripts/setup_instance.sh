#!/usr/bin/env bash
# Provision a bare vast.ai GPU instance for TNF OpenMM MD.
#
# No custom Docker image is needed: a stock CUDA/PyTorch image plus micromamba gets
# OpenMM running in a few minutes, which is far cheaper than building and pushing an
# image (the lesson from the gnina-reinvent build).
#
# Run ON the instance:   bash setup_instance.sh
set -euo pipefail

WORK=/workspace/tnf
MAMBA_ROOT=/opt/micromamba

echo "=== GPU / driver ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
# The driver's CUDA ceiling decides which OpenMM build is loadable. Locally, an
# OpenMM built for cuda-version 12.9 on a 12.8 driver failed every CUDA context with
# CUDA_ERROR_UNSUPPORTED_PTX_VERSION (222). Read it and pin accordingly.
DRIVER_CUDA=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' | head -1)
echo "driver supports CUDA ${DRIVER_CUDA}"

echo "=== micromamba ==="
if [ ! -x "${MAMBA_ROOT}/bin/micromamba" ]; then
  mkdir -p "${MAMBA_ROOT}/bin"
  # The tarball member is literally "bin/micromamba"; do NOT pass
  # --strip-components=1 or it lands at ${MAMBA_ROOT}/micromamba and the
  # bin/ path below is never created.
  curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xj -C "${MAMBA_ROOT}" bin/micromamba
  chmod +x "${MAMBA_ROOT}/bin/micromamba"
fi
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT}"
eval "$("${MAMBA_ROOT}/bin/micromamba" shell hook -s bash)"

# Pin cuda-version to at most the driver's ceiling. 12.6 builds load on any >=12.6
# driver, so it is the safe default; bump only if the driver is older.
CUDA_PIN="12.6"
case "${DRIVER_CUDA}" in
  12.0|12.1|12.2|12.3|12.4|12.5) CUDA_PIN="12.0" ;;
esac
echo "pinning cuda-version=${CUDA_PIN}"

micromamba create -y -n tnfmd -c conda-forge \
  python=3.12 openmm "cuda-version=${CUDA_PIN}" pdbfixer mdtraj numpy
micromamba activate tnfmd

echo "=== CUDA context smoke test ==="
python - <<'PY'
import openmm
print("openmm", openmm.__version__)
p = openmm.Platform.getPlatformByName("CUDA")
s = openmm.System(); s.addParticle(1.0)
f = openmm.CustomExternalForce("x^2"); f.addParticle(0, []); s.addForce(f)
c = openmm.Context(s, openmm.VerletIntegrator(0.001), p)
c.setPositions([[0, 0, 0]])
print("CUDA context OK:", c.getState(getEnergy=True).getPotentialEnergy())
PY

mkdir -p "${WORK}"/{data,src,runs}
echo
echo "=== ready ==="
echo "upload data/*.pdb and src/*.py into ${WORK}, then run run_md.sh"
