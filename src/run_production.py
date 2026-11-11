"""Production MD from an equilibrated TNF system, writing a trajectory for CV training.

Resumes from `equilibrated.xml` written by prep_system.py and runs NVT/NPT production,
saving frames often enough to train a CV on (DeepTDA needs ensembles, not single
structures) and to measure the thermal fluctuation of the candidate geometric CVs.

Usage:
    python src/run_production.py RUN_DIR [--ns 5] [--save-ps 10] [--platform CUDA]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import openmm
import openmm.app as app
from openmm import unit


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--ns", type=float, default=5.0)
    ap.add_argument("--save-ps", type=float, default=10.0, help="frame interval, ps")
    ap.add_argument("--platform", default="CUDA")
    args = ap.parse_args()

    d = args.run_dir
    pdb = app.PDBFile(str(d / "equilibrated.pdb"))
    ff = app.ForceField("amber14-all.xml", "amber14/tip3pfb.xml")
    system = ff.createSystem(pdb.topology, nonbondedMethod=app.PME,
                             nonbondedCutoff=1.0 * unit.nanometer,
                             constraints=app.HBonds)
    system.addForce(openmm.MonteCarloBarostat(1 * unit.bar, 300 * unit.kelvin, 25))

    dt = 2 * unit.femtosecond
    sim = None
    for name in (args.platform, "CPU"):
        integrator = openmm.LangevinMiddleIntegrator(300 * unit.kelvin,
                                                     1 / unit.picosecond, dt)
        try:
            platform = openmm.Platform.getPlatformByName(name)
            sim = app.Simulation(pdb.topology, system, integrator, platform)
            break
        except Exception as exc:  # noqa: BLE001
            # A PTX/driver mismatch only raises when the Context is built.
            print(f"platform {name} unusable: {type(exc).__name__}: {exc}")
    if sim is None:
        raise SystemExit("no usable OpenMM platform")

    state_file = d / "equilibrated.xml"
    if state_file.exists():
        sim.loadState(str(state_file))
    else:
        sim.context.setPositions(pdb.positions)
        sim.context.setVelocitiesToTemperature(300 * unit.kelvin)

    steps = int(args.ns * 1e6 / dt.value_in_unit(unit.femtosecond))
    interval = max(1, int(args.save_ps * 1000 / dt.value_in_unit(unit.femtosecond)))
    sim.reporters.append(app.DCDReporter(str(d / "production.dcd"), interval))
    sim.reporters.append(app.StateDataReporter(
        str(d / "production.log"), interval, step=True, time=True,
        potentialEnergy=True, temperature=True, volume=True, speed=True))

    print(f"{d.name}: {args.ns} ns on {platform.getName()}, "
          f"{steps} steps, frame every {interval} steps ({args.save_ps} ps)")
    t0 = time.time()
    sim.step(steps)
    wall = time.time() - t0

    sim.saveState(str(d / "production_final.xml"))
    report = {"ns": args.ns, "wall_s": round(wall, 1),
              "ns_per_day": round(args.ns / (wall / 86400), 2),
              "platform": platform.getName(),
              "n_frames": steps // interval,
              "frame_interval_ps": args.save_ps}
    (d / "production_report.json").write_text(json.dumps(report, indent=2))
    print(f"done: {report['ns_per_day']} ns/day, {report['n_frames']} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
