"""Each relaxation converges to its hard counterpart in the sharp limit and is differentiable."""

import os
import sys

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ders import relaxations as R  # noqa: E402


def test_spatial_read_limit():
    centers = jp.array([[0.0], [1.0], [2.0]])
    feats = jp.array([[1.0], [5.0], [9.0]])
    q = jp.array([0.8])
    # at a cell center the read equals the nearest-cell lookup
    qc = jp.array([1.0])
    soft = R.triangular_spatial_read(feats, centers, qc, bandwidth=0.5 + 1e-6)
    assert np.allclose(soft, R.hard_spatial_read(feats, centers, qc), atol=1e-6)
    g = jax.grad(lambda qq: R.triangular_spatial_read(feats, centers, qq, 1.5)[0])(q)
    assert np.isfinite(g).all() and abs(float(g[0])) > 0


def test_skill_route_limit():
    logits = jp.array([0.1, 2.0, -1.0])
    outs = jp.array([[1.0], [2.0], [3.0]])
    assert np.allclose(R.skill_route(logits, outs, 1e-3), R.hard_skill_route(logits, outs))


def test_contact_gate_limit():
    d = jp.array([-0.01, 0.0, 0.01])
    assert np.allclose(R.contact_gate(d, 1e4), jp.array([1.0, 0.5, 0.0]), atol=1e-6)


def test_depth_composite_limit():
    depths = jp.array([0.5, 0.2, 0.9])
    attrs = jp.array([[0.0], [1.0], [2.0]])
    assert np.allclose(R.soft_depth_composite(depths, attrs, 1e-3), R.hard_depth_composite(depths, attrs))
