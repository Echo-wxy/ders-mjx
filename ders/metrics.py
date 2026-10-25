"""Jacobian comparison metrics and contact-proximity labels."""

from __future__ import annotations

import mujoco
import numpy as np


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float(a @ b / (na * nb))


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = cosine(a, b)
    return float("nan") if np.isnan(c) else float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def contact_distance(scene, d_mj: mujoco.MjData, distmax: float = 0.5) -> float:
    """phi_min: smallest |signed distance| over the task contact pairs (m).

    Pairs are gripper geom / object geom and object geom / non-support
    environment geom (for example the box with the hole). Support contacts of
    the object with the table are excluded because they are always active.
    """
    pairs = [(g1, g2) for g1 in scene.gripper_geoms for g2 in scene.obj_geoms]
    pairs += [(g1, g2) for g1 in scene.obj_geoms for g2 in scene.env_geoms]
    best = distmax
    for g1, g2 in pairs:
        dist = mujoco.mj_geomDistance(scene.mj, d_mj, int(g1), int(g2), distmax, None)
        best = min(best, abs(dist))
    return float(best)


def active_set(scene, d) -> np.ndarray:
    """Boolean mask of active contacts (signed distance < 0) among candidates that
    involve the object or the gripper. Candidate order is fixed by MJX."""
    geom = np.asarray(d.contact.geom)
    dist = np.asarray(d.contact.dist)
    involved = np.isin(geom, np.r_[scene.obj_geoms, scene.gripper_geoms]).any(axis=1)
    return (dist < 0) & involved


def gripper_object_gap(scene, d_mj: mujoco.MjData, distmax: float = 0.5) -> float:
    """Smallest signed distance between a gripper geom and an object geom (m).

    Positive: separated (the pre-contact band is ``0 < gap <= contact margin``);
    zero or negative: touching or penetrating.
    """
    best = distmax
    for g1 in scene.gripper_geoms:
        for g2 in scene.obj_geoms:
            best = min(best, mujoco.mj_geomDistance(scene.mj, d_mj, int(g1), int(g2), distmax, None))
    return float(best)
