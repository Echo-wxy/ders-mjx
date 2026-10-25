"""The coupled transitions must execute the hard step bit for bit."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
from mujoco import mjx  # noqa: E402

from ders.engine import Scene, make_transition  # noqa: E402
from ders.sampling import ScriptedController  # noqa: E402


def test_bitwise_forward_push():
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
        assert np.array_equal(q.view(np.uint32), ref.view(np.uint32)), m
