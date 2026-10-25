"""Runtime setup shared by the scripts (compilation cache, CPU threads)."""

from __future__ import annotations

import os

import jax


def setup_cache(path: str | None = None) -> str:
    """Enable the persistent XLA compilation cache.

    MJX programs take tens of seconds to compile; the cache makes a second run
    of the same script start immediately. The cache directory defaults to
    ``.jax_cache`` next to the repository root and can be moved with the
    ``DERS_JAX_CACHE`` environment variable.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = path or os.environ.get("DERS_JAX_CACHE", os.path.join(root, ".jax_cache"))
    os.makedirs(path, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", path)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
    return path


def load_tuned(scene: str) -> dict:
    """Relaxation parameters selected on the tuning seed (``configs/tuned.json``)."""
    import json

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "configs", "tuned.json")
    if not os.path.exists(path):
        return {}
    return json.load(open(path)).get(scene, {})
