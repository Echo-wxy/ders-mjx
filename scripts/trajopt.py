"""Multi-step gradients, one-step descent test and short-horizon optimization (G5, G6).

Every method executes the same hard MuJoCo/MJX rollout; only the gradient of
the rollout cost differs:

``hard_ad``  automatic differentiation of the hard step;
``cfd``      Contacts From Distance (Paulus et al.), straight-through;
``ders``     DERS contact gate, straight-through;
``generic``  compliant-contact relaxation, straight-through;
``rs``       zeroth-order randomized smoothing of the rollout cost (antithetic).

Modes
-----
``descent``   G6a and G5. For each problem: gradient at the nominal sequence,
              cost change after one normalized step of several sizes, and
              (``--fd-problems k``) cosine with a central finite-difference
              gradient of the hard rollout cost on the first k problems.
``optimize``  G6b. Adam for ``--iters`` iterations; the full cost curve is
              stored so that equal-iteration and equal-simulation-call
              comparisons can both be read off.
``tune``      Pre-registered grid of relaxation parameters, evaluated with the
              descent test on the tuning seed only (seed 0). Writes the
              selected values to ``configs/tuned.json``; the evaluation seeds
              (1-5) are never used for tuning.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402

# Gradient comparisons (descent, tune, finite differences) run in float64 so
# that finite differences of the rollout cost are not dominated by round-off;
# the optimization runs in float32.
if any(m in sys.argv for m in ("descent", "tune")):
    jax.config.update("jax_enable_x64", True)

import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

from ders.engine import Scene, Temperatures, default_params, make_transition  # noqa: E402
from ders.problems import SUCCESS_RADIUS, make_problems, rollout_cost  # noqa: E402
from ders.runtime import load_tuned, setup_cache  # noqa: E402

ETAS = (0.1, 0.3, 1.0)
GRID = {  # pre-registered candidate values (tuning seed 0 only)
    "ders": {"alpha": [100.0, 300.0, 1000.0], "spring": [0.002, 0.005, 0.01]},
    "generic": {"soften": [3.0, 10.0, 30.0]},
    "cfd": {"cfd_width": [0.01, 0.03, 0.05]},
    "rs": {"sigma": [0.03, 0.1, 0.3]},
    "hard_ad": {},
}
RS_PAIRS = 8
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Estimator:
    """Gradient of the hard rollout cost for one method (compiled once)."""

    def __init__(self, scene, method, reach_weight):
        self.scene, self.method = scene, method
        rw = reach_weight
        self.hard = make_transition(scene, "hard_ad")
        if method == "rs":
            def cost_hard(th, d0, goal):
                return rollout_cost(scene, self.hard, th, d0, goal, reach_weight=rw)[0]

            def grad(th, d0, goal, p, key):
                w = p["sigma"] * jax.random.normal(key, (RS_PAIRS,) + th.shape)
                cp = jax.vmap(lambda wi: cost_hard(th + wi, d0, goal))(w)
                cm = jax.vmap(lambda wi: cost_hard(th - wi, d0, goal))(w)
                return jp.einsum("n,n...->...", cp - cm, w) / (2 * RS_PAIRS * p["sigma"] ** 2)
            self.grad = jax.jit(grad)
            self.calls_per_grad = 2 * RS_PAIRS
        else:
            step = make_transition(scene, method)

            def grad(th, d0, goal, p, key):
                del key
                return jax.grad(lambda t: rollout_cost(scene, step, t, d0, goal, p, rw)[0])(th)
            self.grad = jax.jit(grad)
            self.calls_per_grad = 3  # one forward and one reverse pass, about three forwards
        self.cost = jax.jit(lambda th, d0, goal: rollout_cost(scene, self.hard, th, d0, goal,
                                                              reach_weight=rw))


def params_for(method, tuned, override=None):
    p = default_params(Temperatures())
    p["sigma"] = jp.asarray(0.1)
    for k, v in (tuned.get(method, {}) | (override or {})).items():
        p[k] = jp.asarray(v)
    return p


def descent_rows(est, problems, p, key, label, etas=ETAS):
    rows = []
    for i, pr in enumerate(problems):
        key, sub = jax.random.split(key)
        th0 = pr["theta0"]
        g = est.grad(th0, pr["d0"], pr["goal"], p, sub)
        g = jp.where(jp.isfinite(g), g, 0.0)
        gn = float(jp.linalg.norm(g))
        c0 = float(est.cost(th0, pr["d0"], pr["goal"])[0])
        row = dict(problem=i, method=label, grad_norm=gn, cost0=c0)
        for eta in etas:
            if gn > 0:
                c1 = float(est.cost(th0 - eta * g / gn, pr["d0"], pr["goal"])[0])
            else:
                c1 = c0
            row[f"dcost_eta{eta:g}"] = c1 - c0
        row["grad"] = np.asarray(g)
        rows.append(row)
    return rows


def fd_gradient(est, pr, h):
    th0 = pr["theta0"]
    flat = th0.reshape(-1)
    eye = jp.eye(flat.size)
    cost = jax.jit(jax.vmap(lambda e: est.cost((flat + h * e).reshape(th0.shape), pr["d0"], pr["goal"])[0]
                            - est.cost((flat - h * e).reshape(th0.shape), pr["d0"], pr["goal"])[0]))
    out = []
    for k in range(0, flat.size, 16):
        out.append(np.asarray(cost(eye[k:k + 16])))
    return (np.concatenate(out) / (2 * h)).reshape(th0.shape)


def run_descent(args, scene, problems, tuned):
    rows, fd_store = [], {}
    if args.fd_problems:
        hard_est = Estimator(scene, "hard_ad", problems[0]["reach_weight"])
        for i in range(min(args.fd_problems, len(problems))):
            fd_store[i] = {h: fd_gradient(hard_est, problems[i], h) for h in args.fd_hs}
        print(f"  finite differences on {len(fd_store)} problems done", flush=True)
    for m in args.methods:
        tm = time.time()
        est = Estimator(scene, m, problems[0]["reach_weight"])
        rs = descent_rows(est, problems, params_for(m, tuned), jax.random.PRNGKey(args.seed), m)
        for r in rs:
            g = r.pop("grad")
            for h, gfd in fd_store.get(r["problem"], {}).items():
                den = np.linalg.norm(g) * np.linalg.norm(gfd)
                r[f"cos_fd_h{h:g}"] = float((g * gfd).sum() / den) if den > 0 else float("nan")
        rows += rs
        print(f"  {m}: {time.time() - tm:.0f} s", flush=True)
        del est
        jax.clear_caches()
    return rows


def run_optimize(args, scene, problems, tuned):
    rows = []
    for m in args.methods:
        tm = time.time()
        est = Estimator(scene, m, problems[0]["reach_weight"])
        p = params_for(m, tuned)
        key = jax.random.PRNGKey(args.seed)
        for i, pr in enumerate(problems):
            th = pr["theta0"]
            mom, vel = jp.zeros_like(th), jp.zeros_like(th)
            curve = [float(est.cost(th, pr["d0"], pr["goal"])[0])]
            for it in range(1, args.iters + 1):
                key, sub = jax.random.split(key)
                g = est.grad(th, pr["d0"], pr["goal"], p, sub)
                g = jp.where(jp.isfinite(g), g, 0.0)
                mom = 0.9 * mom + 0.1 * g
                vel = 0.999 * vel + 0.001 * g ** 2
                th = th - args.lr * (mom / (1 - 0.9 ** it)) / (jp.sqrt(vel / (1 - 0.999 ** it)) + 1e-8)
                curve.append(float(est.cost(th, pr["d0"], pr["goal"])[0]))
            c_fin, obj = est.cost(th, pr["d0"], pr["goal"])
            dist = float(jp.linalg.norm(obj - pr["goal"]))
            rows.append(dict(problem=i, method=m, cost0=curve[0], cost_final=float(c_fin),
                             best_cost=min(curve), final_dist=dist, success=int(dist < SUCCESS_RADIUS),
                             calls_per_iter=est.calls_per_grad,
                             curve=";".join(f"{c:.6g}" for c in curve)))
        print(f"  {m}: success {np.mean([r['success'] for r in rows if r['method'] == m]):.2f}, "
              f"{time.time() - tm:.0f} s", flush=True)
        del est
        jax.clear_caches()
    return rows


def run_tune(args, scene, problems):
    rows, best = [], {}
    for m in args.methods:
        grid = GRID[m]
        est = Estimator(scene, m, problems[0]["reach_weight"])
        import itertools
        keys = list(grid)
        combos = [{}] if not grid else [dict(zip(keys, vals)) for vals in itertools.product(*[grid[k] for k in keys])]
        scores = []
        for combo in combos:
            label = m + "".join(f"|{k}={v:g}" for k, v in combo.items())
            rs = descent_rows(est, problems, params_for(m, {}, combo), jax.random.PRNGKey(0), label)
            for r in rs:
                r.pop("grad")
            rows += rs
            # pre-registered criterion: mean cost change over the three step sizes
            scores.append(np.mean([np.mean([r[f"dcost_eta{e:g}"] for e in ETAS]) for r in rs]))
            print(f"  {label}: mean cost change {scores[-1]:.3e}", flush=True)
        # ties (within 0.1 %) are broken in favour of the default value
        smin = min(scores)
        near = [i for i, sc in enumerate(scores) if sc <= smin + 1e-3 * abs(smin)]
        defaults = {"alpha": 300.0, "spring": 0.005, "soften": 10.0, "cfd_width": 0.01, "sigma": 0.1}
        pick = next((i for i in near if all(defaults.get(k) == v for k, v in combos[i].items())), near[0])
        best[m] = combos[pick]
        del est
        jax.clear_caches()
    return rows, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["descent", "optimize", "tune"], required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--methods", nargs="+", default=["hard_ad", "cfd", "ders", "generic", "rs"])
    p.add_argument("--n-problems", type=int, default=8)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--variant", choices=["nominal", "rest"], default="nominal")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--fd-problems", type=int, default=0)
    p.add_argument("--fd-hs", type=float, nargs="+", default=[1e-2, 1e-3])
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_cache()

    t0 = time.time()
    scene = Scene(args.scene)
    hard = jax.jit(make_transition(scene, "hard_ad"))
    problems = make_problems(scene, hard, args.seed, args.n_problems, args.horizon, args.variant)
    print(f"[{args.scene} seed {args.seed} {args.mode}] {len(problems)} problems, {time.time() - t0:.0f} s",
          flush=True)
    tuned = load_tuned(args.scene).get(args.variant, {})
    if args.mode == "descent":
        rows = run_descent(args, scene, problems, tuned)
    elif args.mode == "optimize":
        rows = run_optimize(args, scene, problems, tuned)
    else:
        rows, best = run_tune(args, scene, problems)
        path = os.path.join(ROOT, "configs", "tuned.json")
        allbest = json.load(open(path)) if os.path.exists(path) else {}
        allbest.setdefault(args.scene, {}).setdefault(args.variant, {}).update(best)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        json.dump(allbest, open(path, "w"), indent=2)
        print(f"selected {best} -> {path}", flush=True)

    for r in rows:
        r.update(scene=args.scene, seed=args.seed, horizon=args.horizon, variant=args.variant)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != "scene", k != "seed", k))
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
