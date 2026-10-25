"""Residual network of the backward model (article Eqs. 3 and 12).

The residual predicts a correction to the rate of change of the state and is
added to the relaxed engine's prediction in the backward model only; the
executed transition is unaffected. Its training target (Eq. 12) is the gap
between the recorded rate of change and the rate predicted by the relaxed
engine, the sum of the relaxation error and of the unmodeled physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jp


def init_mlp(key, sizes):
    params = []
    for k, (a, b) in zip(jax.random.split(key, len(sizes) - 1), zip(sizes[:-1], sizes[1:])):
        w = jax.random.normal(k, (a, b)) / jp.sqrt(a)
        params.append((w, jp.zeros(b)))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def residual_rate_target(x_next, x, x_relaxed_next, dt):
    """Eq. 12: observed rate minus the relaxed engine's predicted rate."""
    return (x_next - x) / dt - (x_relaxed_next - x) / dt


def residual_loss(params, inputs, targets):
    return jp.mean((mlp(params, inputs) - targets) ** 2)


def train_residual(params, inputs, targets, steps: int = 2000, lr: float = 1e-3):
    """Full-batch Adam, kept dependency-free."""
    grad = jax.jit(jax.grad(residual_loss))
    m = jax.tree.map(jp.zeros_like, params)
    v = jax.tree.map(jp.zeros_like, params)
    for t in range(1, steps + 1):
        g = grad(params, inputs, targets)
        m = jax.tree.map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree.map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        params = jax.tree.map(
            lambda p, mm, vv: p - lr * (mm / (1 - 0.9 ** t)) / (jp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8),
            params, m, v)
    return params
