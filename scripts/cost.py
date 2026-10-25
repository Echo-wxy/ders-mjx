"""Computational cost (G7): compile time, steady-state forward and forward+backward time,
scaling with rollout horizon and with the number of batched environments, peak memory.

Timing follows the JAX guidance: the first call (compile + run) is reported
separately, every measurement waits for the result with ``block_until_ready``,
and steady-state numbers are the median over repeated calls.

Run this script alone on an otherwise idle machine::

    python scripts/cost.py --scene push --out results/cost_push.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

from ders.engine import Scene, make_transition  # noqa: E402
from ders.problems import make_problems, rollout_cost  # noqa: E402
from ders.runtime import setup_cache  # noqa: E402


def peak_memory_gb() -> float:
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or getattr(info, "peak_rss", None)
        if peak:
            return peak / 2 ** 30
        return info.rss / 2 ** 30
    except ImportError:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def timed(fn, *args, repeats=5):
    t = time.perf_counter()
    jax.block_until_ready(fn(*args))
    first = time.perf_counter() - t
    runs = []
    for _ in range(repeats):
        t = time.perf_counter()
        jax.block_until_ready(fn(*args))
        runs.append(time.perf_counter() - t)
    return first, float(np.median(runs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="push")
    p.add_argument("--methods", nargs="+", default=["hard_ad", "cfd", "ders", "generic"])
    p.add_argument("--horizons", type=int, nargs="+", default=[10, 30])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_cache()
    scene = Scene(args.scene)
    hard = jax.jit(make_transition(scene, "hard_ad"))
    pr = make_problems(scene, hard, 1, 1, max(args.horizons))[0]
    rows = []
    meta = dict(scene=args.scene, cpu=platform.processor() or platform.machine(),
                platform=platform.platform(), jax=jax.__version__, device=str(jax.devices()[0]))

    for m in args.methods:
        step = make_transition(scene, m)
        # single forward step
        f1 = jax.jit(lambda d, a: step(d, a).qpos)
        first, steady = timed(f1, pr["d0"], jp.zeros(4), repeats=args.repeats)
        rows.append(dict(meta, method=m, measure="forward_step", horizon=1, batch=1,
                         first_call_s=first, steady_s=steady, steps_per_s=1.0 / steady))
        for H in args.horizons:
            th = pr["theta0"][:H]
            fwd = jax.jit(lambda t: rollout_cost(scene, step, t, pr["d0"], pr["goal"])[0])
            fb = jax.jit(jax.value_and_grad(lambda t: rollout_cost(scene, step, t, pr["d0"], pr["goal"])[0]))
            first_f, steady_f = timed(fwd, th, repeats=args.repeats)
            first_b, steady_b = timed(fb, th, repeats=args.repeats)
            rows.append(dict(meta, method=m, measure="rollout_forward", horizon=H, batch=1,
                             first_call_s=first_f, steady_s=steady_f, steps_per_s=H / steady_f))
            rows.append(dict(meta, method=m, measure="rollout_forward_backward", horizon=H, batch=1,
                             first_call_s=first_b, steady_s=steady_b, steps_per_s=H / steady_b))
            print(f"[{m}] H={H}: forward {steady_f:.3f} s, forward+backward {steady_b:.3f} s", flush=True)
        for B in args.batches:
            dB = jax.tree.map(lambda x: jp.stack([x] * B), pr["d0"])
            fB = jax.jit(jax.vmap(lambda d, a: step(d, a).qpos))
            first, steady = timed(fB, dB, jp.zeros((B, 4)), repeats=args.repeats)
            rows.append(dict(meta, method=m, measure="batched_forward_step", horizon=1, batch=B,
                             first_call_s=first, steady_s=steady, steps_per_s=B / steady))
        rows[-1]["peak_memory_gb"] = peak_memory_gb()
        print(f"[{m}] done, peak memory so far {peak_memory_gb():.2f} GB", flush=True)
        jax.clear_caches()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = sorted({k for r in rows for k in r})
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
