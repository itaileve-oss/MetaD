# Multiple-walker WT metadynamics image: OpenMM 8.4 + PLUMED (conda-forge) + openmm-plumed.
#   docker pull ghcr.io/itaileve-oss/metad:latest
#   docker run --gpus all -it -v $PWD:/workspace ghcr.io/itaileve-oss/metad:latest
# CUDA user-space libraries come from conda-forge (cuda-version 12.6), so any NVIDIA driver
# with CUDA >= 12.6 works; no nvidia/cuda base image is needed.
FROM mambaorg/micromamba:1.5.10

USER root
RUN apt-get update && apt-get install -y --no-install-recommends procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*
# launch scripts use /opt/micromamba/envs/tnfmd (the layout on our GPU boxes)
RUN ln -s /opt/conda /opt/micromamba

RUN micromamba create -y -n tnfmd -c conda-forge \
        python=3.12 openmm=8.4.0 cuda-version=12.6 pdbfixer=1.12 mdtraj=1.11.2 \
        plumed openmm-plumed=2.1 py-plumed numpy scipy pandas matplotlib \
    && micromamba clean -afy

COPY . /opt/metad
ENV PATH=/opt/conda/envs/tnfmd/bin:$PATH

# build-time smoke test (no GPU needed)
RUN python -c "import openmm, openmmplumed, pdbfixer, mdtraj; print('openmm', openmm.__version__, 'openmm-plumed OK')" \
    && plumed info --version

WORKDIR /workspace
CMD ["/bin/bash"]
