"""Straight-through coupling between the hard and the soft transition.

The executed value is the hard value itself; the derivative is that of the
soft value. Implemented with a custom JVP so that no floating-point arithmetic
touches the hard value (see ``soft_contact._exact``).
"""

import jax

from .soft_contact import _exact


def straight_through(hard, soft):
    """Return ``hard`` bit for bit, with the derivative of ``soft``."""
    return jax.tree.map(_exact, hard, soft)
