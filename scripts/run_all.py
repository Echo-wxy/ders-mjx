"""Run every experiment of the revision in order. Resumable: a job whose output
file already exists is skipped, so the script can simply be started again after
an interruption.

Stages
------
1 tune     relaxation parameters on the tuning seed 0 (sequential)
2 warmup   compile every program once, one job at a time (keeps peak memory low;
           later jobs load the compiled programs from the cache)
3 main     G1 forward consistency, G3 gradient fidelity, G4 temperature sweep,
           G5 multi-step gradients, G6a one-step descent, G6b optimization
           (``--jobs`` processes in parallel)
4 cost     G7 computational cost (alone)
5 round    pre-contact band experiment on round-geometry scenes (``--stage round``)

Usage::

    python scripts/run_all.py --jobs 4
    python scripts/run_all.py --stage main --jobs 4     # one stage only
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
PY = sys.executable
GRAD_SCENES = ["reach", "push", "plate_slide", "pick_place", "peg_insert"]
OPT_SCENES = ["push", "plate_slide", "pick_place"]
VARIANTS = ["nominal", "rest"]
SEEDS = [1, 2, 3, 4, 5]
ALPHA_SCALES = ["0.25", "0.5", "1", "2", "4"]


def job(script, out, *args):
    return dict(cmd=[PY, os.path.join(ROOT, "scripts", script), *map(str, args), "--out", out], out=out)


def plan():
    stages = {"tune": [], "warmup": [], "main": [], "cost": []}
    for s in OPT_SCENES:
        for v in VARIANTS:
            stages["tune"].append(job("trajopt.py", f"{RES}/tune/tune_{s}_{v}.csv", "--mode", "tune",
                                      "--scene", s, "--seed", 0, "--variant", v, "--n-problems", 4))
    W = f"{RES}/warmup"
    for s in GRAD_SCENES:
        stages["warmup"].append(job("grad_fidelity.py", f"{W}/g3_{s}.npz", "--scene", s, "--seeds", 1,
                                    "--max-states", 1))
        stages["warmup"].append(job("forward.py", f"{W}/g1_{s}.csv", "--scene", s, "--n-states", 1,
                                    "--steps", 2))
    for s in ("push", "peg_insert"):
        stages["warmup"].append(job("grad_fidelity.py", f"{W}/g4_{s}.npz", "--scene", s, "--seeds", 1,
                                    "--max-states", 1, "--methods", "hard_ad", "--alpha-scales", *ALPHA_SCALES))
    for s in OPT_SCENES:
        for H, fd in ((30, 1), (5, 1)):
            stages["warmup"].append(job("trajopt.py", f"{W}/g6a_{s}_H{H}.csv", "--mode", "descent", "--scene", s,
                                        "--seed", 1, "--n-problems", 1, "--horizon", H, "--fd-problems", fd))
        stages["warmup"].append(job("trajopt.py", f"{W}/g6b_{s}.csv", "--mode", "optimize", "--scene", s,
                                    "--seed", 1, "--n-problems", 1, "--iters", 1))

    M = RES
    for s in OPT_SCENES:  # longest jobs first
        for v in VARIANTS:
            for seed in SEEDS:
                stages["main"].append(job("trajopt.py", f"{M}/g6b/g6b_{s}_{v}_s{seed}.csv", "--mode", "optimize",
                                          "--scene", s, "--variant", v, "--seed", seed))
    for s in GRAD_SCENES:
        stages["main"].append(job("grad_fidelity.py", f"{M}/g3/g3_{s}.npz", "--scene", s))
        stages["main"].append(job("forward.py", f"{M}/g1/g1_{s}.csv", "--scene", s))
    for s in ("push", "peg_insert"):
        stages["main"].append(job("grad_fidelity.py", f"{M}/g4/g4_{s}.npz", "--scene", s, "--seeds", 1, 2, 3,
                                  "--methods", "hard_ad", "--alpha-scales", *ALPHA_SCALES))
    for s in OPT_SCENES:
        for v in VARIANTS:
            for seed in SEEDS:
                stages["main"].append(job("trajopt.py", f"{M}/g6a/g6a_{s}_{v}_s{seed}.csv", "--mode", "descent",
                                          "--scene", s, "--variant", v, "--seed", seed, "--fd-problems", 2))
        stages["main"].append(job("trajopt.py", f"{M}/g5/g5_{s}_H5.csv", "--mode", "descent", "--scene", s,
                                  "--seed", 1, "--horizon", 5, "--fd-problems", 8))
    stages["cost"].append(job("cost.py", f"{M}/g7/cost_push.csv", "--scene", "push"))

    # round: pre-contact band experiment (G8). Finger pads as capsules and
    # objects as spheres/capsules, the only MJX collision pairs that report a
    # separation distance, so that backward relaxations can act before contact.
    # All five methods, same protocol as the main runs.
    R = f"{RES}/round"
    RS = ["push_round", "plate_slide_round"]
    stages["round_tune"] = [job("trajopt.py", f"{R}/tune/tune_{s}_{v}.csv", "--mode", "tune", "--scene", s,
                                "--seed", 0, "--variant", v, "--n-problems", 4) for s in RS for v in VARIANTS]
    stages["round_warmup"] = []
    for s in RS:
        stages["round_warmup"].append(job("grad_fidelity.py", f"{R}/warmup/g3_{s}.npz", "--scene", s, "--seeds", 1,
                                          "--max-states", 1))
        stages["round_warmup"].append(job("trajopt.py", f"{R}/warmup/g6b_{s}.csv", "--mode", "optimize", "--scene", s,
                                          "--seed", 1, "--n-problems", 1, "--iters", 1))
        stages["round_warmup"].append(job("trajopt.py", f"{R}/warmup/g6a_{s}.csv", "--mode", "descent", "--scene", s,
                                          "--seed", 1, "--n-problems", 1, "--fd-problems", 1))
    stages["round_main"] = []
    for s in RS:
        for v in VARIANTS:
            for seed in SEEDS:
                stages["round_main"].append(job("trajopt.py", f"{R}/g6b/g6b_{s}_{v}_s{seed}.csv", "--mode",
                                                "optimize", "--scene", s, "--variant", v, "--seed", seed))
    for s in RS:
        stages["round_main"].append(job("grad_fidelity.py", f"{R}/g3/g3_{s}.npz", "--scene", s))
        stages["round_main"].append(job("forward.py", f"{R}/g1/g1_{s}.csv", "--scene", s))
    for s in RS:
        for v in VARIANTS:
            for seed in SEEDS:
                stages["round_main"].append(job("trajopt.py", f"{R}/g6a/g6a_{s}_{v}_s{seed}.csv", "--mode", "descent",
                                                "--scene", s, "--variant", v, "--seed", seed, "--fd-problems", 2))
    return stages


def run_one(j):
    if os.path.exists(j["out"]):
        return j["out"], "skipped (exists)", 0.0
    os.makedirs(os.path.dirname(j["out"]), exist_ok=True)
    logdir = os.path.join(RES, "logs")
    os.makedirs(logdir, exist_ok=True)
    log = os.path.join(logdir, os.path.basename(j["out"]) + ".log")
    t = time.time()
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    return j["out"], "ok" if rc == 0 else f"FAILED (exit {rc}, see {log})", time.time() - t


class _Tee:
    def __init__(self, path):
        self.fh = open(path, "a", encoding="utf-8")

    def write(self, s):
        sys.__stdout__.write(s)
        self.fh.write(s)
        self.fh.flush()

    def flush(self):
        sys.__stdout__.flush()


def main():
    os.makedirs(RES, exist_ok=True)
    sys.stdout = _Tee(os.path.join(RES, "run_all_log.txt"))
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["all", "tune", "warmup", "main", "cost", "round"], default="all")
    p.add_argument("--jobs", type=int, default=4)
    args = p.parse_args()
    stages = plan()
    if args.stage == "all":
        order = ["tune", "warmup", "main", "cost"]
    elif args.stage == "round":
        order = ["round_tune", "round_warmup", "round_main"]
    else:
        order = [args.stage]
    t0 = time.time()
    failed = []
    for st in order:
        n = args.jobs if st in ("main", "round_main") else 1
        print(f"=== stage {st}: {len(stages[st])} jobs, {n} in parallel", flush=True)
        with ThreadPoolExecutor(max_workers=n) as ex:
            for k, (out, status, sec) in enumerate(ex.map(run_one, stages[st]), 1):
                print(f"[{time.strftime('%H:%M:%S')}] {st} {k}/{len(stages[st])} {os.path.relpath(out, ROOT)}: "
                      f"{status} ({sec / 60:.1f} min, total {(time.time() - t0) / 3600:.2f} h)", flush=True)
                if status.startswith("FAILED"):
                    failed.append(out)
    print(f"finished in {(time.time() - t0) / 3600:.2f} h; failed jobs: {len(failed)}", flush=True)
    for f in failed:
        print("  FAILED", f)


if __name__ == "__main__":
    main()
