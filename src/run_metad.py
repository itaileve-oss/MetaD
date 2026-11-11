"""Run WT-metadynamics (via PLUMED) from an equilibrated TNF system.

Also serves as the plain unbiased runner (--no-bias), which is how the CV fluctuations
are measured to pick METAD's SIGMA, and how the A/B trajectories needed by Gen-COMPAS
are produced.

Protocol follows bioRxiv 10.1101/2025.11.28.690881 (ff14SB/TIP3P/150 mM KCl, 310 K,
4 fs + HMR, 12 A cutoff switching from 10 A). NOTE their biasing method was meta-eABF;
PLUMED's DRR module was unavailable here, so this is WT-metaD on the same CVs. State
that whenever the free energies are compared.

Usage:
    python src/run_metad.py RUN_DIR --ns 250 --plumed RUN_DIR/plumed.dat [--walker-id 0]
    python src/run_metad.py RUN_DIR --ns 5 --no-bias --out-prefix unbiased
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import openmm
import openmm.app as app
from openmm import unit


def build(run_dir: Path, args):
    pdb = app.PDBFile(str(run_dir / args.state_pdb))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")
    system = ff.createSystem(
        pdb.topology, nonbondedMethod=app.PME,
        nonbondedCutoff=args.cutoff * unit.nanometer,
        switchDistance=args.switch * unit.nanometer,
        constraints=app.HBonds, rigidWater=True,
        hydrogenMass=(args.hmr * unit.amu) if args.hmr else None,
    )
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, args.temp * unit.kelvin, 25))

    if not args.no_bias:
        from openmmplumed import PlumedForce
        script = Path(args.plumed).read_text()
        system.addForce(PlumedForce(script))
    return pdb, system


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ns", type=float, required=True)
    ap.add_argument("--plumed", default=None)
    ap.add_argument("--no-bias", action="store_true")
    ap.add_argument("--out-prefix", default="metad")
    # Trajectory size is the real constraint, not compute. A full-system frame of this
    # TNF box is ~800 kB (69k atoms incl. water), so 250 ns at 20 ps x 4 walkers would be
    # ~40 GB against 46 GB free. Protein-only frames are ~10x smaller, and 100 ps is
    # ample for the slow collective motion being sampled.
    ap.add_argument("--save-ps", type=float, default=100.0)
    ap.add_argument("--all-atoms", action="store_true",
                    help="write water too; off by default to keep DCDs ~10x smaller")
    ap.add_argument("--temp", type=float, default=310.0)
    ap.add_argument("--timestep-fs", type=float, default=4.0)
    ap.add_argument("--hmr", type=float, default=1.5)
    ap.add_argument("--cutoff", type=float, default=1.2)
    ap.add_argument("--switch", type=float, default=1.0)
    ap.add_argument("--platform", default="CUDA")
    # Both phases must run from ONE directory so they append to the same HILLS (PLUMED
    # resolves FILE= against the process cwd). They start from different systems, so the
    # starting state has to be selectable rather than fixed to run_dir/equilibrated.*.
    ap.add_argument("--state-pdb", default="equilibrated.pdb")
    ap.add_argument("--state-xml", default="equilibrated.xml")
    ap.add_argument("--checkpoint-ns", type=float, default=10.0,
                    help="write a resumable checkpoint this often")
    ap.add_argument("--resume", action="store_true",
                    help="continue from <prefix>.chk if present (use with a plumed.dat "
                         "generated with --restart so hills are appended)")
    ap.add_argument("--allow-cpu", action="store_true",
                    help="permit the CPU fallback; off by default because a silent "
                         "fallback wasted 20 minutes of wall clock once already")
    args = ap.parse_args()
    if not args.no_bias and not args.plumed:
        raise SystemExit("--plumed is required unless --no-bias")

    d = args.run_dir
    pdb, system = build(d, args)
    dt = args.timestep_fs * unit.femtosecond

    sim = None
    names = [args.platform] + (["CPU"] if args.allow_cpu else [])
    for name in names:
        integrator = openmm.LangevinMiddleIntegrator(
            args.temp * unit.kelvin, 1 / unit.picosecond, dt)
        try:
            platform = openmm.Platform.getPlatformByName(name)
            sim = app.Simulation(pdb.topology, system, integrator, platform)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"platform {name} unusable: {type(exc).__name__}: {exc}")
    if sim is None:
        raise SystemExit(
            "Simulation could not be built. Two common causes, which look alike because "
            "both surface when the Context is created:\n"
            "  1. a PLUMED script error (check the assertion text printed above by "
            "PLUMED itself -- it aborts before OpenMM gets a chance);\n"
            "  2. a PTX/driver mismatch (CUDA_ERROR_UNSUPPORTED_PTX_VERSION), meaning "
            "conda's cuda-version exceeds the driver's ceiling -- re-pin it rather than "
            "falling back to CPU, which is ~100x slower here."
        )

    prefix = args.out_prefix
    chk = d / f"{prefix}.chk"
    if args.resume and chk.exists():
        sim.loadCheckpoint(str(chk))
        print(f"resumed from {chk}", flush=True)
        state_file = None
    else:
        state_file = d / args.state_xml
    if state_file is not None and state_file.exists():
        sim.loadState(str(state_file))
    else:
        sim.context.setPositions(pdb.positions)
        sim.context.setVelocitiesToTemperature(args.temp * unit.kelvin)

    steps = int(args.ns * 1e6 / args.timestep_fs)
    interval = max(1, int(args.save_ps * 1000 / args.timestep_fs))
    if args.all_atoms:
        sim.reporters.append(app.DCDReporter(str(d / f"{prefix}.dcd"), interval))
        n_written = pdb.topology.getNumAtoms()
    else:
        # openmm's own DCDReporter cannot subset atoms; mdtraj's can.
        from mdtraj.reporters import DCDReporter as MDTrajDCD
        solvent = {"HOH", "WAT", "NA", "CL", "K", "MG", "ZN"}
        keep = [a.index for a in pdb.topology.atoms()
                if a.residue.name.strip() not in solvent]
        sim.reporters.append(MDTrajDCD(str(d / f"{prefix}.dcd"), interval,
                                       atomSubset=keep))
        n_written = len(keep)
        # A protein-only trajectory needs its own topology file to be readable later.
        sel = app.Modeller(pdb.topology, pdb.positions)
        sel.delete([r for r in pdb.topology.residues()
                    if r.name.strip() in solvent])
        with (d / f"{prefix}_top.pdb").open("w") as fh:
            app.PDBFile.writeFile(sel.topology, sel.positions, fh)
    sim.reporters.append(app.StateDataReporter(
        str(d / f"{prefix}.log"), interval, step=True, time=True,
        potentialEnergy=True, temperature=True, volume=True, speed=True))
    # Checkpoints make the run safe to stop at any moment: HILLS is already on disk, so
    # the FES earned so far is never lost, and the trajectory can be continued later.
    chk_steps = max(1, int(args.checkpoint_ns * 1e6 / args.timestep_fs))
    sim.reporters.append(app.CheckpointReporter(str(chk), chk_steps))

    mode = "unbiased" if args.no_bias else "WT-metaD"
    est_mb = (steps / interval) * n_written * 12 / 1e6
    print(f"{d.name}: {mode}, {args.ns} ns on {platform.getName()}, {steps} steps, "
          f"frame every {args.save_ps} ps, {n_written} atoms/frame "
          f"(~{est_mb:.0f} MB trajectory)", flush=True)
    t0 = time.time()
    sim.step(steps)
    wall = time.time() - t0

    sim.saveState(str(d / f"{prefix}_final.xml"))
    report = {"mode": mode, "ns": args.ns, "wall_s": round(wall, 1),
              "ns_per_day": round(args.ns / (wall / 86400), 2),
              "platform": platform.getName(), "temp_K": args.temp,
              "timestep_fs": args.timestep_fs, "hydrogen_mass_amu": args.hmr,
              "n_frames": steps // interval,
              "atoms_per_frame": n_written,
              "traj_mb_est": round((steps / interval) * n_written * 12 / 1e6, 1)}
    (d / f"{prefix}_report.json").write_text(json.dumps(report, indent=2))
    print(f"done: {report['ns_per_day']} ns/day, {report['n_frames']} frames", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
