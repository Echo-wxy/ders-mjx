"""Environment self-check (about 10 minutes). Writes results/selfcheck.txt.

1. imports and versions;
2. the five scenes load and the scripted controller reaches its contacts;
3. the coupled transitions execute the hard step bit for bit;
4. one reverse-mode gradient of a 30-step rollout: compile and steady-state time,
   used to estimate the total running time of the experiments.
"""

from __future__ import annotations

import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lines = []


def log(msg):
    print(msg, flush=True)
    lines.append(msg)


def main():
    t0 = time.time()
    import jax
    import jax.numpy as jp
    import mujoco
    import numpy as np
    from mujoco import mjx
    import softjax  # noqa: F401

    from ders.engine import SCENES, Scene, make_transition
    from ders.metrics import contact_distance
    from ders.problems import make_problems, rollout_cost
    from ders.runtime import setup_cache
    from ders.sampling import ScriptedController

    setup_cache()
    log(f"python {platform.python_version()} | {platform.platform()} | cpu count {os.cpu_count()}")
    log(f"jax {jax.__version__} devices {jax.devices()} | mujoco {mujoco.__version__}")

    for name in SCENES:
        sc = Scene(name)
        hard = jax.jit(make_transition(sc, "hard_ad"))
        rng = np.random.default_rng(0)
        d = mjx.put_data(sc.mj, sc.reset_data(rng))
        ctrl = ScriptedController(sc, rng)
        o0 = np.asarray(d.xpos[sc.obj_id]).copy()
        phis = []
        for _ in range(150):
            d = hard(d, jp.asarray(ctrl(d), dtype=d.qpos.dtype))
            phis.append(contact_distance(sc, mjx.get_data(sc.mj, d)))
        moved = float(np.linalg.norm(np.asarray(d.xpos[sc.obj_id]) - o0))
        log(f"scene {name:12s} min contact distance {min(phis):.4f} m, object moved {moved:.3f} m")

    sc = Scene("push")
    hard = jax.jit(make_transition(sc, "hard_ad"))
    rng = np.random.default_rng(1)
    d = mjx.put_data(sc.mj, sc.reset_data(rng))
    ctrl = ScriptedController(sc, rng)
    for _ in range(40):
        d = hard(d, jp.asarray(ctrl(d), dtype=d.qpos.dtype))
    a = jp.asarray(ctrl(d), dtype=d.qpos.dtype)
    ref = np.asarray(hard(d, a).qpos)
    for m in ("cfd", "ders", "generic"):
        q = np.asarray(jax.jit(make_transition(sc, m))(d, a).qpos)
        log(f"bitwise forward {m:8s}: {np.array_equal(q.view(np.uint32), ref.view(np.uint32))}")

    pr = make_problems(sc, hard, 1, 1, 30)[0]
    step = make_transition(sc, "ders")
    g = jax.jit(jax.grad(lambda t: rollout_cost(sc, step, t, pr["d0"], pr["goal"])[0]))
    t = time.time()
    jax.block_until_ready(g(pr["theta0"]))
    compile_s = time.time() - t
    t = time.time()
    for _ in range(3):
        jax.block_until_ready(g(pr["theta0"]))
    grad_s = (time.time() - t) / 3
    log(f"DERS 30-step gradient: first call {compile_s:.0f} s, steady {grad_s:.2f} s")
    # optimization stage dominates: 3 scenes x 2 variants x 5 seeds x 8 problems x 5 methods x 50 iterations
    opt_h = 3 * 2 * 5 * 8 * 5 * 50 * grad_s * 1.15 / 3600
    log(f"estimated optimization stage: {opt_h:.1f} h on one process, about {opt_h / 4:.1f} h with 4 processes")
    log(f"self-check finished in {time.time() - t0:.0f} s: OK")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # report the failure in the log file as well
        import traceback

        lines.append("FAILED: " + repr(e))
        lines.append(traceback.format_exc())
        print(traceback.format_exc())
        raise
    finally:
        os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
        with open(os.path.join(ROOT, "results", "selfcheck.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")
