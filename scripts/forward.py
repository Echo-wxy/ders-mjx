"""Forward consistency of the coupled transition with the hard step (G1).

For every scene, start state and method the script replays the same action
stream (recorded from a closed-loop scripted rollout of the hard step) and
compares the executed states with the hard rollout:

* per-step check: the coupled step and the hard step are evaluated from the
  same hard state, so mismatches cannot accumulate;
* rollout check: the coupled transition is iterated on its own states.

Each check is run with the rounding-free coupling (``exact``) and with the
left-to-right form (``rounding``). Executing the relaxed model itself
(``soft_only``) measures how far a relaxed engine drifts from the hard one.
A bitwise match compares the raw float bits of qpos, qvel and the
object pose.

Usage::

    python scripts/forward.py --scene push --out results/forward_push.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

from ders.engine import Scene, make_transition  # noqa: E402
from ders.runtime import setup_cache  # noqa: E402
from ders.sampling import ScriptedController  # noqa: E402


def state_vec(d):
    return jp.concatenate([d.qpos, d.qvel])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--n-states", type=int, default=10)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--methods", nargs="+", default=["cfd", "ders", "generic"])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_cache()
    t0 = time.time()
    scene = Scene(args.scene)
    hard = jax.jit(make_transition(scene, "hard_ad"))
    hard_again = jax.jit(make_transition(scene, "hard_ad"))

    # closed-loop hard rollouts define the start states and action streams
    streams = []
    for k in range(args.n_states):
        rng = np.random.default_rng(500 + k)
        d = mjx.put_data(scene.mj, scene.reset_data(rng))
        ctrl = ScriptedController(scene, rng)
        states, actions = [d], []
        for _ in range(args.steps):
            a = jp.asarray(ctrl(states[-1]), dtype=d.qpos.dtype)
            actions.append(a)
            states.append(hard(states[-1], a))
        streams.append((states, actions))
    print(f"[{args.scene}] {args.n_states} hard rollouts, {time.time() - t0:.0f} s", flush=True)

    rows = []
    configs = [("hard_ad", "repeat")]
    configs += [(m, f) for m in args.methods for f in ("exact", "rounding", "soft_only")]
    for method, form in configs:
        tm = time.time()
        if form == "repeat":
            step = hard_again
        else:
            step = jax.jit(make_transition(scene, method, coupling=form))
        for k, (states, actions) in enumerate(streams):
            ref = [np.asarray(state_vec(s)) for s in states[1:]]
            # per-step check from the hard states
            per = [np.asarray(state_vec(step(states[t], actions[t]))) for t in range(args.steps)]
            per_bit = np.array([np.array_equal(x.view(np.uint32 if x.dtype == np.float32 else np.uint64),
                                               r.view(np.uint32 if r.dtype == np.float32 else np.uint64))
                                for x, r in zip(per, ref)])
            per_err = np.array([np.linalg.norm(x - r) / max(np.linalg.norm(r), 1e-12) for x, r in zip(per, ref)])
            # rollout check
            d, roll = states[0], []
            for t in range(args.steps):
                d = step(d, actions[t])
                roll.append(np.asarray(state_vec(d)))
            roll_bit = np.array([np.array_equal(x, r) and np.array_equal(np.signbit(x), np.signbit(r))
                                 for x, r in zip(roll, ref)])
            roll_err = np.array([np.linalg.norm(x - r) / max(np.linalg.norm(r), 1e-12) for x, r in zip(roll, ref)])
            obj_drift = float(np.linalg.norm(np.asarray(d.xpos[scene.obj_id]) - np.asarray(states[-1].xpos[scene.obj_id])))
            mism = per_err[~per_bit]
            rows.append(dict(
                scene=args.scene, state=k, method=method, coupling=form, steps=args.steps,
                step_bitwise_frac=float(per_bit.mean()),
                step_mean_rel_err=float(per_err.mean()),
                step_mismatch_mean_rel_err=float(mism.mean()) if mism.size else 0.0,
                step_mismatch_max_rel_err=float(mism.max()) if mism.size else 0.0,
                rollout_bitwise_frac=float(roll_bit.mean()),
                rollout_mean_rel_err=float(roll_err.mean()),
                rollout_final_rel_err=float(roll_err[-1]),
                rollout_first_mismatch=int(np.argmin(roll_bit)) if (~roll_bit).any() else -1,
                obj_final_drift_m=obj_drift,
            ))
        print(f"[{args.scene}] {method}/{form}: step bitwise "
              f"{np.mean([r['step_bitwise_frac'] for r in rows[-len(streams):]]):.6f}, "
              f"rollout bitwise {np.mean([r['rollout_bitwise_frac'] for r in rows[-len(streams):]]):.6f}, "
              f"{time.time() - tm:.0f} s", flush=True)
        jax.clear_caches()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
