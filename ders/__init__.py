"""DERS: dual-execution simulation on MuJoCo XLA.

The executed transition is always the hard MuJoCo/MJX step. Gradients come
from a separately relaxed backward model selected by a gradient estimator.
"""

__version__ = "0.1.0"
