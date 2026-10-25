"""Relaxed contact rows used by the soft (backward) model.

The MJX fork used here (the ``diffmjx`` branch of MuJoCo) already evaluates a
hard and a soft forward pass in every substep and joins them with a
straight-through estimator whenever ``opt.cfd_enable`` is set. Its soft branch
implements Contacts From Distance (CFD). This module replaces the pyramidal
contact rows of that soft branch so that the relaxation can be selected:

``cfd``   the unchanged CFD rows of the fork (baseline, Paulus et al.);
``gate``    the DERS contact gate (manuscript Eq. 9): the hard activation
            indicator of each candidate in the contact band is replaced by
            ``sigmoid(-alpha * dist)``, which scales its constraint row;
``soften``  a conventional compliant-contact relaxation: the activation stays
            the hard indicator and the contact time constant is multiplied
            by ``factor`` (generic straight-through baseline).

Only the soft branch is changed; the hard rows are those of MuJoCo, so the
executed (forward) transition is unaffected by the choice.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import jax
import jax.numpy as jp
import numpy as np
from mujoco.mjx._src import constraint as _c
from mujoco.mjx._src import support

_ORIGINAL_PYRAMIDAL = _c._efc_contact_pyramidal
_STATE = {"mode": "cfd", "alpha": 500.0}


@contextlib.contextmanager
def coupling_form(form: str):
    """Select the coupling while tracing.

    ``exact``      sg(hard) + (soft - sg(soft)): forward value is the hard state;
    ``rounding``   sg(hard) + soft - sg(soft) evaluated left to right (rounds twice);
    ``soft_only``  the relaxed model is executed (no coupling), for drift measurements.
    """
    if form not in ("exact", "rounding", "soft_only"):
        raise ValueError(form)
    old = COUPLING["form"]
    COUPLING["form"] = form
    try:
        yield
    finally:
        COUPLING["form"] = old


@contextlib.contextmanager
def soft_mode(mode: str, alpha=500.0):
    """Select the soft contact relaxation while a function is being traced.

    ``alpha`` is the gate sharpness (1/m) for ``gate`` and the time-constant
    factor for ``soften``. It may be a traced JAX value, so one compiled
    program serves every temperature of a sweep.
    """
    if mode not in ("cfd", "gate", "soften"):
        raise ValueError(f"unknown soft contact mode {mode!r}")
    old = dict(_STATE)
    _STATE.update(mode=mode, alpha=alpha)  # alpha may be a traced value
    try:
        yield
    finally:
        _STATE.clear()
        _STATE.update(old)


def _gated_pyramidal(m, d, condim: int, soft: bool) -> Optional[_c._Efc]:
    if not soft or _STATE["mode"] == "cfd":
        return _ORIGINAL_PYRAMIDAL(m, d, condim, soft)

    con_id = np.nonzero(d.contact.dim == condim)[0]
    if con_id.size == 0:
        return None
    alpha = _STATE["alpha"]

    @jax.vmap
    def rows(c):
        pos_aref = c.dist - c.includemargin
        solref = c.solref

        body1, body2 = jp.array(m.geom_bodyid)[c.geom]
        jac1p, jac1r = support.jac(m, d, c.pos, body1)
        jac2p, jac2r = support.jac(m, d, c.pos, body2)
        diff = c.frame @ (jac2p - jac1p).T
        if condim > 3:
            diff = jp.concatenate((diff, (c.frame @ (jac2r - jac1r).T)), axis=0)
        fri = jp.repeat(c.friction[: condim - 1], 2, axis=0).at[1::2].mul(-1)
        j = diff[0] + jp.repeat(diff[1:condim], 2, axis=0) * fri[:, None]

        invweight = m.body_invweight0[body1, 0] + m.body_invweight0[body2, 0]
        invweight = invweight + fri[0] * fri[0] * invweight
        invweight = invweight * 2 * fri[0] * fri[0] / m.opt.impratio

        if _STATE["mode"] == "gate":
            # DERS contact gate (manuscript Eq. 9): the hard activation
            # indicator 1[dist <= 0] of every candidate in the contact band is
            # replaced by g = sigmoid(-alpha * dist), which scales the
            # candidate's constraint row (Jacobian and reference position).
            # A separated candidate therefore exerts a force only through the
            # velocity term of its reference acceleration, i.e. when the bodies
            # approach; as alpha grows the gate tends to the hard indicator
            # away from the contact boundary.
            gate = jax.nn.sigmoid(-alpha * pos_aref)
            return _c._row(
                j * gate,
                pos_aref * gate,
                pos_aref,
                invweight,
                solref,
                c.solimp,
                c.solimp[1],
                c.includemargin,
                jp.zeros_like(pos_aref),
            )
        # soften: hard activation indicator, contact time constant enlarged
        active = (pos_aref < 0).astype(pos_aref.dtype)
        solref = solref.at[0].mul(alpha)
        return _c._row(
            j * active,
            pos_aref * active,
            pos_aref,
            invweight,
            solref,
            c.solimp,
            c.solimp[1],
            c.includemargin,
            jp.zeros_like(pos_aref),
        )

    contact = jax.tree_util.tree_map(lambda x: x[con_id], d.contact)
    return jax.tree_util.tree_map(jp.concatenate, rows(contact))


# Installed once on import; the hard branch and the ``cfd`` mode delegate to
# the original implementation.
_c._efc_contact_pyramidal = _gated_pyramidal


# ----------------------------------------------------------------------------
# Rounding-free straight-through coupling inside every substep.
#
# The fork joins the two passes as ``sg(hard) + soft - sg(soft)``. Python
# evaluates this left to right, i.e. ``(sg(hard) + soft) - sg(soft)``, which
# rounds twice, so the executed state can differ from the hard state in the
# last bits; through stiff contact dynamics such differences grow. The
# ``exact`` coupling returns the hard value itself and attaches the soft
# tangent through a custom JVP, so the executed state is the hard state bit
# for bit by construction. ``COUPLING`` selects the form so that the effect can
# be measured (forward-consistency experiment).
# ----------------------------------------------------------------------------
from mujoco.mjx._src import forward as _f

COUPLING = {"form": "exact"}  # "exact", "rounding" or "soft_only"


@jax.custom_jvp
def _ste(hard, soft):
    """Primal value: ``hard`` (returned unchanged, no arithmetic); tangent: that of ``soft``."""
    return hard


@_ste.defjvp
def _ste_jvp(primals, tangents):
    hard, _ = primals
    _, dsoft = tangents
    return hard, dsoft


def _exact(h, s):
    if jp.issubdtype(h.dtype, jp.floating):
        return _ste(jax.lax.stop_gradient(h), s)
    return h


def _coupled_forward(m, d):
    if not m.opt.cfd_enable:
        return _f._forward(m, d, soft=False)
    d_soft = _f._forward(m, d, soft=True)
    d_hard = _f._forward(m, d, soft=False)
    sg = jax.lax.stop_gradient
    if COUPLING["form"] == "soft_only":
        return d_soft
    if COUPLING["form"] == "exact":
        # Returning the hard value itself guarantees bitwise identity: the
        # arithmetic form sg(h) + (s - sg(s)) is exact only if the compiler
        # evaluates both copies of s identically, which XLA does not promise
        # (fusion and FMA contraction can differ between the two uses).
        fn = _exact
    else:
        fn = lambda h, s: sg(h) + s - sg(s)
    return jax.tree.map(fn, d_hard, d_soft)


_f.forward = _coupled_forward


# ----------------------------------------------------------------------------
# Equality constraints: restore upstream MuJoCo behaviour.
#
# The fork clamps the reference position of *every* constraint row to be
# non-positive (``pos_aref = -relu(-pos_aref)``) inside ``make_constraint``.
# That is intended for contact rows (a separated contact must not pull), but it
# also zeroes the positive half of equality residuals, so the mocap weld that
# drives the Meta-World hand loses half of its restoring force and the hand no
# longer tracks the mocap target. Here the clamp is applied to contact rows
# only, which leaves every contact row exactly as in the fork and makes the
# weld rows match MuJoCo C.
# ----------------------------------------------------------------------------
import mujoco as _mujoco
from mujoco.mjx._src.types import DisableBit as _DisableBit, ConeType as _ConeType


def _clamp_rows(efc):
    if efc is None:
        return None
    return efc.replace(pos_aref=-jax.nn.relu(-efc.pos_aref))


def _make_constraint(m, d, soft: bool):
    if m.opt.disableflags & _DisableBit.CONSTRAINT:
        efcs = ()
    else:
        efcs = (
            _c._efc_equality_connect(m, d),
            _c._efc_equality_weld(m, d),
            _c._efc_equality_joint(m, d),
            _c._efc_equality_tendon(m, d),
            _c._efc_friction(m, d),
            _c._efc_limit_ball(m, d, soft=soft),
            _c._efc_limit_slide_hinge(m, d, soft=soft),
            _c._efc_limit_tendon(m, d, soft=soft),
            _clamp_rows(_c._efc_contact_frictionless(m, d, soft=soft)),
        )
        con_fn = _c._efc_contact_elliptic if m.opt.cone == _ConeType.ELLIPTIC else _c._efc_contact_pyramidal
        efcs += tuple(_clamp_rows(con_fn(m, d, dim, soft=soft)) for dim in (3, 4, 6))
        efcs = tuple(efc for efc in efcs if efc is not None)

    if not efcs:
        z = jp.empty(0)
        d = d.replace(efc_J=jp.empty((0, m.nv)))
        d = d.replace(efc_D=z, efc_aref=z, efc_pos=z, efc_margin=z)
        return d

    efc = jax.tree_util.tree_map(lambda *x: jp.concatenate(x), *efcs)

    @jax.vmap
    def fn(efc):
        k, b, imp = _c._kbi(m, efc.solref, efc.solimp, efc._dmax, efc.pos_imp)
        r = jp.maximum(efc.invweight * (1 - imp) / imp, _mujoco.mjMINVAL)
        aref = -b * (efc.J @ d.qvel) - k * imp * efc.pos_aref
        return aref, r, efc.pos_aref + efc.margin, efc.margin, efc.frictionloss

    aref, r, pos, margin, frictionloss = fn(efc)
    d = d.replace(efc_J=efc.J, efc_D=1 / r, efc_aref=aref, efc_pos=pos, efc_margin=margin)
    d = d.replace(efc_frictionloss=frictionloss)
    return d


_c.make_constraint = _make_constraint
