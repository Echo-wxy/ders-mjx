"""Trajectory problems shared by the multi-step gradient, descent and optimization runs.

A problem is a start state reached by the scripted controller shortly before
the first contact with the object, a goal displaced from the object, and a
nominal open-loop action sequence (the controller's own next actions). All
methods are evaluated on the same problems, so results can be paired.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

from .metrics import contact_distance
from .sampling import ScriptedController

GOAL_OFFSET = {
    "push": np.array([0.0, 0.10, 0.0]),
    "plate_slide": np.array([0.0, 0.10, 0.0]),
    "pick_place": np.array([0.0, 0.0, 0.10]),
    "peg_insert": np.array([0.0, 0.0, 0.10]),
}
SUCCESS_RADIUS = 0.02  # m: final object-to-goal distance counted as success
REACH_WEIGHT = 0.1     # weight of the tool-to-object shaping term
ACTION_LIMIT = 0.95


def to_theta(a):
    """Inverse of the action parameterization ``a = 0.95 tanh(theta)``."""
    return jp.arctanh(jp.clip(a / ACTION_LIMIT, -0.999, 0.999))


VARIANTS = {
    # nominal: start 2 cm before contact, initial guess = the scripted
    #          controller's own next actions, cost with tool-object shaping
    "nominal": dict(approach=0.02, min_gap=0.0, reach_weight=REACH_WEIGHT, from_rest=False),
    # rest:    start inside the 1 cm contact band, separated from the object,
    #          initial guess = hold still, object-only cost (tests whether a
    #          method supplies a useful gradient before the contact is made)
    "rest": dict(approach=0.008, min_gap=0.003, reach_weight=0.0, from_rest=True),
}


def make_problems(scene, hard, seed: int, n: int, horizon: int, variant: str = "nominal",
                  max_steps: int = 300):
    """Return a list of dicts with ``d0``, ``goal``, ``theta0`` (horizon x 4), ``reach_weight``."""
    cfg = VARIANTS[variant]
    approach = cfg["approach"]
    rng = np.random.default_rng(10_000 + seed)
    out, attempts = [], 0
    while len(out) < n and attempts < 20 * n:
        attempts += 1
        d = mjx.put_data(scene.mj, scene.reset_data(rng))
        ctrl = ScriptedController(scene, rng)
        found = False
        for _ in range(max_steps):
            phi = contact_distance(scene, mjx.get_data(scene.mj, d))
            if phi < approach:
                found = phi > cfg["min_gap"]  # rest: separated but inside the contact band
                break
            d = hard(d, jp.asarray(ctrl(d), dtype=d.qpos.dtype))
        if not found:
            continue
        d0, actions, dd = d, [], d
        if cfg["from_rest"]:
            grip = float(ctrl(d)[3])
            actions = [jp.asarray([0.0, 0.0, 0.0, grip], dtype=d.qpos.dtype)] * horizon
        else:  # the controller's next actions from this state
            for _ in range(horizon):
                a = jp.asarray(ctrl(dd), dtype=dd.qpos.dtype)
                actions.append(a)
                dd = hard(dd, a)
        goal = np.asarray(d0.xpos[scene.obj_id]) + GOAL_OFFSET[scene.spec.name]
        out.append(dict(d0=d0, goal=jp.asarray(goal, dtype=d0.qpos.dtype),
                        theta0=to_theta(jp.stack(actions)), reach_weight=cfg["reach_weight"]))
    if len(out) < n:
        raise RuntimeError(f"{scene.spec.name}: only {len(out)} of {n} problems found")
    return out


def rollout_cost(scene, step, theta, d0, goal, params=None, reach_weight=REACH_WEIGHT):
    """Mean running cost of the rollout and the final object position."""
    actions = ACTION_LIMIT * jp.tanh(theta)

    def body(d, a):
        d = step(d, a, params)
        f = scene.features(d)
        tcp, obj = f[0:3], f[3:6]
        c = jp.sum((obj - goal) ** 2) + reach_weight * jp.sum((tcp - obj) ** 2)
        return d, (c, obj)

    _, (costs, objs) = jax.lax.scan(body, d0, actions)
    return jp.mean(costs), objs[-1]
