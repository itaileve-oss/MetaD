# Multiple-walker well-tempered metadynamics with OpenMM + PLUMED

A small, end-to-end pipeline for protein conformational free energies with
well-tempered (WT) multiple-walker metadynamics, using OpenMM as the MD engine and PLUMED
for the bias (via `openmm-plumed`). It covers system preparation, staged equilibration,
CV construction **with automatic verification against reference structures**, the
metaD runs themselves, and FES / reweighting / convergence analysis.

The worked example is the **TNF-α trimer**: symmetric apo state (PDB 1TNF) vs the
asymmetric, inhibitor-stabilized state (PDB 7JRA, ligand removed). See
[`examples/tnf/`](examples/tnf/), including an honest account of what did and did not work.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/setup_instance.sh` | Provision a stock CUDA GPU box (vast.ai-style) with micromamba + OpenMM; reads the driver's CUDA ceiling and pins a compatible OpenMM build |
| 0 | `scripts/install_plumed.sh`, `scripts/build_plumed.sh` | PLUMED 2.9 + `openmm-plumed` |
| 1 | `src/prep_system.py` | Clean, rebuild missing atoms/loops (PDBFixer), protonate, solvate (TIP3P, 150 mM KCl), ff14SB, HMR |
| 2 | `src/equil_protocol.py` | Staged, restrained ~9 ns equilibration (heat, water NPT, stepwise release, free NPT) so the starting conformation survives preparation |
| 3 | `src/make_plumed.py` | Build the PLUMED CV groups (center-of-mass distances) and **verify** them by recomputing the CVs on the prepared coordinates against crystal reference values; aborts on mismatch (catches residue-numbering and chain-mapping errors) |
| 3b | `src/make_plumed_s.py` | Variant adding a salt-bridge "switch" CV `s` (smooth-minimum contact sums) |
| 4 | `src/run_metad.py` | WT-metaD (or unbiased MD with `--no-bias`) with checkpoints, multiple walkers sharing a HILLS directory |
| 4 | `scripts/metad_v2_walkers.sh`, `scripts/metad_v3_launch.sh` | Two-walker launchers (one walker started from each state) |
| 5 | `src/fes.py` | `sum_hills` FES at any time during the run + block-based convergence check |
| 5 | `analysis/fes_reweight.py` | Python hill sum, final-bias reweighting onto unbiased CVs, window dG between states, dG(t) |
| 5 | `src/analyze_cvs.py` | CV time series / basin widths from trajectories (used to choose `SIGMA`) |
| 5 | `src/check_pbc_cvs.py` | Independent recomputation of the CVs with explicit PBC handling vs PLUMED's COLVAR |
| 5 | `src/saltbridge_cv.py` | Salt-bridge switch CV on trajectories (PBC-safe) |
| 5 | `analysis/plot_fes_metad_v2.py` | FES plot with state boxes and visited-cell masking |

## Quick start with Docker (everything preinstalled)

```bash
docker pull ghcr.io/itaileve-oss/metad:latest
docker run --gpus all -it -v $PWD:/workspace ghcr.io/itaileve-oss/metad:latest
```

The image is built automatically from this repo by GitHub Actions. It contains OpenMM 8.4
(CUDA 12.6), PDBFixer, mdtraj, PLUMED with `openmm-plumed`, and this repository in
`/opt/metad`, with the environment at `/opt/micromamba/envs/tnfmd`. It needs an NVIDIA driver
supporting CUDA ≥ 12.6.

**On vast.ai:** create a template with image `ghcr.io/itaileve-oss/metad:latest` (launch mode
SSH), start an instance, put your inputs in `/workspace`, and run the scripts. No setup needed.

## Requirements

- OpenMM ≥ 8.1 (CUDA), PDBFixer, `openmm-plumed`, PLUMED 2.9
- Python: numpy, scipy, pandas, mdtraj, matplotlib

## Lessons learned (worth reading before using this on a new system)

1. **Bias the CV that has to move first.** On TNF the S3 interface distance must open
   before S1 and the salt bridges can switch (we learned this from the Gen-COMPAS
   minimum free-energy path). Biasing S1 + salt-bridge switch while leaving S3 unbiased
   (v3) gave walkers that moved along the biased CVs but never crossed in S3: two
   disconnected islands and a meaningless dG.
2. **Two islands, each filled by "its" walker, give an arbitrary dG.** In metaD, the
   relative height of two basins is only meaningful after the same walker has crossed
   back and forth. Always plot dG(t); if it swings by several kcal/mol, there is no
   result yet.
3. **Put walls on every CV that can run away.** In v2 the B walker escaped to very high
   S3 and never came back.
4. **Verify CV groups numerically** against reference structures (crystal values) before
   any production run. PDB numbering schemes (precursor vs mature) and PDBFixer
   renumbering are easy to get wrong.
5. **Staged restrained equilibration matters.** A short unrestrained equilibration
   collapsed the asymmetric state before production started.

## License

MIT. See `LICENSE`.
