"""Mechanism-specific relaxations of the backward model (article Eqs. 7-11).

Each function implements one relaxation as a differentiable JAX operation.
The forward (executed) value of every mechanism is the hard one; these
relaxations enter only the backward model and are joined to the hard value by
``ders.coupling.straight_through``. The contact gate is applied to the constraint rows inside the MJX constraint
solver by ``ders.soft_contact``; the stand-alone functions here are the reference
definitions and are used by the unit tests.
"""

from __future__ import annotations

import jax
import jax.numpy as jp


def triangular_spatial_read(features, cell_centers, query, bandwidth, eps: float = 1e-12):
    """Normalized triangular-kernel read (Eq. 7).

    ``features``: (cells, channels); ``cell_centers``: (cells, dims);
    ``query``: (dims,). On a regular grid of spacing h in D dimensions every
    query lies within h*sqrt(D)/2 of its nearest cell, so a bandwidth larger
    than h*sqrt(D)/2 guarantees a positive normalizer. The read coincides with
    the nearest-cell lookup at cell centers and interpolates between
    neighbouring cells elsewhere.
    """
    dist = jp.linalg.norm(cell_centers - query[None, :], axis=-1)
    w = jp.clip(1.0 - dist / bandwidth, 0.0, None)
    w = w / jp.maximum(w.sum(), eps)
    return (w[:, None] * features).sum(0)


def hard_spatial_read(features, cell_centers, query):
    """Nearest-cell read: the executed counterpart of Eq. 7."""
    dist = jp.linalg.norm(cell_centers - query[None, :], axis=-1)
    return features[jp.argmin(dist)]


def skill_route(logits, skill_outputs, temperature):
    """Softmax skill router (Eq. 8): blend of the K skill outputs."""
    w = jax.nn.softmax(logits / temperature)
    return (w[:, None] * skill_outputs).sum(0)


def hard_skill_route(logits, skill_outputs):
    """Arg-max selection: the executed counterpart of Eq. 8."""
    return skill_outputs[jp.argmax(logits)]


def contact_gate(signed_distance, alpha):
    """Logistic gate on the signed distance (Eq. 9); alpha in 1/m."""
    return jax.nn.sigmoid(-alpha * signed_distance)


def gated_contact_impulse(signed_distance, candidate_impulse, alpha):
    """Gated impulse of the backward model (Eq. 10): each candidate's impulse
    weighted by its gate. In the MJX implementation (``ders.soft_contact``)
    the gate scales the candidate's constraint row before the solve."""
    return (contact_gate(signed_distance, alpha)[:, None] * candidate_impulse).sum(0)


def soft_depth_composite(depths, attributes, temperature):
    """Depth-weighted soft rasterization of one pixel (Eq. 11)."""
    w = jax.nn.softmax(-depths / temperature)
    return (w[:, None] * attributes).sum(0)


def hard_depth_composite(depths, attributes):
    """Nearest-face selection: the executed counterpart of Eq. 11."""
    return attributes[jp.argmin(depths)]
