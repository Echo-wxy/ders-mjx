"""Single-step gradient fidelity against central finite differences of the hard step.

For every evaluation state the script stores, in one ``.npz`` file per scene:

* the action Jacobian of every gradient estimator (``J_<method>``, N x 12 x 4);
* central finite-difference Jacobians of the hard step for every step size
  (``FD_<h>``) together with the active contact sets on both sides of every
  perturbation (``switch_<h>``: True where some perturbation changes the
  active set of the next state, i.e. a mode-switch step);
* contact labels: ``phi`` (smallest |signed distance| of the task contact
  pairs), ``gap`` (smallest signed gripper-object distance) and ``active``
  (active contact set, from which the analysis derives set changes).

Metrics (cosine, angle, RMSE, sign flips, near/far/switch strata) are computed
afterwards by ``scripts/analyze.py`` from these files, so the run itself never
has to be repeated to change an analysis choice.

Usage (one scene per process)::

    python scripts/grad_fidelity.py --scene push --out results/grad_push.npz
    python scripts/grad_fidelity.py --scene push --alpha-scales 0.1 0.3 1 3 10 \
        --out results/temp_push.npz          # temperature sweep (G4)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

from ders.engine import Scene, Temperatures, default_params, make_transition  # noqa: E402
from ders.metrics import active_set, contact_distance, gripper_object_gap  # noqa: E402
from ders.runtime import load_tuned, setup_cache  # noqa: E402
from ders.sampling import sample_states  # noqa: E402

DEFAULT_H = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 1e-5, 1e-6)
DEFAULT_SIGMA = (0.01, 0.03, 0.1)


def build(scene: Scene, methods, temps: Temperatures, hs, sigmas, n_rs: int):
    hard = make_transition(scene, "hard_ad", temps)

    def feat_and_active(d, a):
        dn = hard(d, a)
        return scene.features(dn), dn.contact.dist < 0

    jac_fns = {}
    for m in methods:
        step = make_transition(scene, m, temps)
        jac_fns[m] = jax.jit(jax.jacfwd(lambda d, a, step=step: scene.features(step(d, a)), argnums=1))

    hs_arr = jp.asarray(hs)
    sig_arr = jp.asarray(sigmas)
    nh, ns = len(hs), len(sigmas)
    # all finite-difference perturbations in one batch: (h, column, sign)
    pert = (hs_arr[:, None, None, None] * jp.eye(4)[None, :, None, :]
            * jp.array([1.0, -1.0])[None, None, :, None]).reshape(-1, 4)
    batched = jax.vmap(feat_and_active, in_axes=(None, 0))

    @jax.jit
    def fd_all(d, a):
        f, act = batched(d, a[None] + pert)
        f = f.reshape(nh, 4, 2, -1)
        act = act.reshape(nh, 4, 2, -1)
        J = jp.swapaxes((f[:, :, 0] - f[:, :, 1]) / (2.0 * hs_arr[:, None, None]), 1, 2)
        switch = jp.any(act[:, :, 0] != act[:, :, 1], axis=(1, 2))
        return J, switch

    @jax.jit
    def rs_all(d, a, key):
        w = jax.random.normal(key, (n_rs, 4))[None] * sig_arr[:, None, None]  # (ns, n, 4)
        wf = w.reshape(-1, 4)
        fp, _ = batched(d, a[None] + wf)
        fm, _ = batched(d, a[None] - wf)
        df = (fp - fm).reshape(ns, n_rs, -1)
        return jp.einsum("snf,snk->sfk", df, w) / (2.0 * n_rs * sig_arr[:, None, None] ** 2)

    return jax.jit(hard), jac_fns, fd_all, rs_all


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--n-steps", type=int, default=200)
    p.add_argument("--methods", nargs="+", default=["hard_ad", "cfd", "ders", "generic"])
    p.add_argument("--hs", type=float, nargs="+", default=list(DEFAULT_H))
    p.add_argument("--sigmas", type=float, nargs="+", default=list(DEFAULT_SIGMA))
    p.add_argument("--n-rs", type=int, default=32)
    p.add_argument("--alpha-scales", type=float, nargs="*", default=[],
                   help="temperature sweep: DERS gate sharpness multiplied by these factors")
    p.add_argument("--max-states", type=int, default=0, help="debug: limit states per seed")
    p.add_argument("--stride", type=int, default=1, help="debug: keep every k-th state")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_cache()

    scene = Scene(args.scene)
    # pre-registered rule: every scene uses the values selected on push (nominal, seed 0)
    tuned = load_tuned("push").get("nominal", {})
    base = Temperatures(alpha_ders=tuned.get("ders", {}).get("alpha", Temperatures.alpha_ders),
                        soften_factor=tuned.get("generic", {}).get("soften", Temperatures.soften_factor),
                        cfd_width=tuned.get("cfd", {}).get("cfd_width", Temperatures.cfd_width))
    print(f"[{args.scene}] relaxation parameters: {base}", flush=True)
    methods = list(args.methods)
    t0 = time.time()
    hard, jac_fns, fd_all, rs_all = build(scene, methods, base, args.hs, args.sigmas, args.n_rs)
    # temperature sweep: one compiled DERS program, gate sharpness passed as data
    variants = {f"ders_x{k:g}": dict(default_params(base), alpha=jp.asarray(base.alpha_ders * k))
                for k in args.alpha_scales}
    var_fns = {}
    if variants:
        ders_step = make_transition(scene, "ders", base)
        jac_v = jax.jit(jax.jacfwd(lambda d, a, p: scene.features(ders_step(d, a, p)), argnums=1))
        fwd_v = jax.jit(lambda d, a, p: ders_step(d, a, p).qpos)
        var_fns = {v: (lambda d, a, p=p: jac_v(d, a, p), lambda d, a, p=p: fwd_v(d, a, p))
                   for v, p in variants.items()}

    # Pass 1: evaluation states, contact labels, finite differences, randomized smoothing.
    rec = {k: [] for k in ["seed", "t", "phi", "gap", "active", "fd", "switch", "rs"]}
    states = []
    for seed in args.seeds:
        traj = sample_states(scene, hard, seed=seed, n_steps=args.n_steps)
        idx = list(range(0, len(traj), args.stride))
        if args.max_states:
            idx = idx[: args.max_states]
        key = jax.random.PRNGKey(seed)
        for t in idx:
            d, a = traj[t]
            key, sub = jax.random.split(key)
            rec["seed"].append(seed)
            rec["t"].append(t)
            d_mj = mjx.get_data(scene.mj, d)
            rec["phi"].append(contact_distance(scene, d_mj))
            rec["gap"].append(gripper_object_gap(scene, d_mj))
            rec["active"].append(active_set(scene, d))
            J, sw = fd_all(d, a)
            rec["fd"].append(np.asarray(J))
            rec["switch"].append(np.asarray(sw))
            rec["rs"].append(np.asarray(rs_all(d, a, sub)))
            states.append((d, a))
        print(f"[{args.scene}] seed {seed}: {len(idx)} states labelled, {time.time() - t0:.0f} s", flush=True)
    del fd_all, rs_all
    jax.clear_caches()

    # Pass 2: one estimator at a time (keeps peak memory low).
    for m in methods:
        fn = jac_fns.pop(m)
        rec[f"J_{m}"] = [np.asarray(fn(d, a)) for d, a in states]
        del fn
        jax.clear_caches()
        print(f"[{args.scene}] {m}: done, {time.time() - t0:.0f} s", flush=True)
    for v in list(var_fns):
        jf, ff = var_fns[v]
        rec[f"J_{v}"] = []
        rec[f"fwd_bitwise_{v}"] = []
        for d, a in states:
            rec[f"J_{v}"].append(np.asarray(jf(d, a)))
            q_hard = np.asarray(hard(d, a).qpos)
            q = np.asarray(ff(d, a))
            rec[f"fwd_bitwise_{v}"].append(np.array_equal(q.view(np.uint64), q_hard.view(np.uint64)))
        print(f"[{args.scene}] {v}: done, {time.time() - t0:.0f} s", flush=True)

    out = {k: np.asarray(v) for k, v in rec.items()}
    out["hs"] = np.asarray(args.hs)
    out["sigmas"] = np.asarray(args.sigmas)
    out["methods"] = np.asarray(methods)
    out["variants"] = np.asarray(list(variants))
    out["seconds"] = np.asarray(time.time() - t0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez_compressed(args.out, **out)
    print(f"saved {args.out} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
