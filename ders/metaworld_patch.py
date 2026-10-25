"""Load Meta-World v3 Sawyer scenes and make their collision geometry MJX-compatible.

The JAX implementation of MJX supports a subset of collision pairs and no
margins for meshes. The patch keeps the visual geometry untouched and changes
collision geometry only:

* robot links other than the gripper claws and pads stop colliding;
* collidable meshes and cylinders that belong to task objects are replaced by
  boxes with the same local bounding box;
* every collidable geom gets ``margin = gap = contact_margin``, so contact
  candidates exist within ``contact_margin`` of touching while the hard engine
  activates a contact only at zero signed distance (MuJoCo activates a contact
  when ``dist < margin - gap``). With the default 1 cm the executed trajectory
  is identical to the one with zero margin (checked in MuJoCo C); larger
  margins change the box-box contact points and are therefore not used;
* the mocap weld is reset as in Meta-World (the hand tracks the mocap body);
* the friction cone is pyramidal and the solver iteration counts are fixed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

GRIPPER_GEOMS = ("rightclaw_it", "rightpad_geom", "leftclaw_it", "leftpad_geom")
ROBOT_BODY_PREFIXES = (
    "right_", "head", "screen", "pedestal", "torso", "controller_box", "hand",
)
REPLACEABLE = (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_CYLINDER)


@dataclass(frozen=True)
class PatchConfig:
    pads: str = "box"             # "box" (Meta-World) or "capsules": finger pads and claws as
                                  # three parallel capsules; MJX's box-box collider reports no
                                  # separation distance, capsule colliders do
    round_objects: bool = False   # object geoms as spheres/capsules (separation distances are
                                  # only reported by MJX's sphere/capsule colliders)
    contact_margin: float = 0.01
    solver_iterations: int = 6
    ls_iterations: int = 6
    timestep: float = 0.0025


@dataclass
class PatchReport:
    disabled: list = field(default_factory=list)
    replaced: list = field(default_factory=list)
    collidable: list = field(default_factory=list)


def metaworld_asset_dir() -> str:
    """Directory of the Meta-World Sawyer XML files.

    Only the assets of the ``metaworld`` package are used. The package is
    located without importing it (its ``__init__`` imports gymnasium, which is
    not needed here).
    """
    import importlib.util

    spec = importlib.util.find_spec("metaworld")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("metaworld is not installed: pip install --no-deps metaworld")
    return os.path.join(list(spec.submodule_search_locations)[0], "assets", "sawyer_xyz")


def _is_robot_body(name: str) -> bool:
    return bool(name) and name.startswith(ROBOT_BODY_PREFIXES)


def _pads_to_capsules(spec, ref, report):
    """Replace each pad/claw box (half sizes hx, hy, hz) by three capsules of
    radius hy running along x, stacked along z, on the same body."""
    bodies = {b.name: b for b in spec.bodies}
    for g in list(spec.geoms):
        if g.name not in GRIPPER_GEOMS:
            continue
        hx, hy, hz = [float(v) for v in g.size[:3]]
        gid = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_GEOM, g.name)
        body = bodies[mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, ref.geom_bodyid[gid])]
        base_pos = np.asarray(g.pos, dtype=float)
        # capsule axis is z in the geom frame; rotate z onto x: quat (w, x, y, z) = (c, 0, s, 0)
        quat = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
        offsets = [-hz + hy, 0.0, hz - hy]
        for k, dz in enumerate(offsets):
            if k == 1:
                geom = g
            else:
                geom = body.add_geom()
                geom.name = f"{g.name}_cap{k}"
                geom.contype, geom.conaffinity = g.contype, g.conaffinity
                geom.condim = g.condim
                geom.friction = g.friction
                geom.solref = g.solref
                geom.solimp = g.solimp
                geom.rgba = g.rgba
                geom.group = g.group
            geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
            geom.size = [hy, hx - hy, 0.0]
            geom.pos = (base_pos + np.array([0.0, 0.0, dz])).tolist()
            geom.quat = quat
            geom.margin = 0.0
            geom.gap = 0.0
        report.replaced.append(f"{g.name} -> 3 capsules")


ROUND_OBJECTS = {  # geom name -> ("sphere", radius) or ("capsule_x", radius, half_length)
    "objGeom": ("sphere", 0.02),
    "puck": ("capsule_y", 0.015, 0.035),  # plate: capsule along the push direction, same height
    "peg": ("capsule_x", 0.015, 0.06),
}


def _round_objects(spec, ref, report):
    for g in spec.geoms:
        if g.name in ROUND_OBJECTS:
            shape = ROUND_OBJECTS[g.name]
            if shape[0] == "sphere":
                g.type = mujoco.mjtGeom.mjGEOM_SPHERE
                g.size = [shape[1], 0.0, 0.0]
            else:
                g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
                g.size = [shape[1], shape[2], 0.0]
                if shape[0] == "capsule_x":    # capsule axis z -> x
                    g.quat = [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0]
                elif shape[0] == "capsule_y":  # capsule axis z -> y
                    g.quat = [np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0]
            g.meshname = ""
            report.replaced.append(f"{g.name} -> {shape[0]}")


def load_patched_model(xml_name: str, cfg: PatchConfig = PatchConfig()):
    """Return ``(mj_model, report)`` for a Meta-World XML such as ``sawyer_push_v3.xml``."""
    path = os.path.join(metaworld_asset_dir(), xml_name)
    spec = mujoco.MjSpec.from_file(path)
    ref = spec.compile()  # compiled once to read geom bounding boxes
    report = PatchReport()
    for i, g in enumerate(spec.geoms):
        if not (g.contype or g.conaffinity):
            continue
        body = mujoco.mj_id2name(ref, mujoco.mjtObj.mjOBJ_BODY, ref.geom_bodyid[i]) or ""
        label = g.name or f"geom{i}@{body}"
        if g.name in GRIPPER_GEOMS:
            pass
        elif _is_robot_body(body):
            g.contype, g.conaffinity = 0, 0
            report.disabled.append(label)
            continue
        elif g.type in REPLACEABLE:
            # geom_aabb stores (center, half-size) in the geom frame
            center, half = ref.geom_aabb[i, :3], ref.geom_aabb[i, 3:]
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = half.tolist()
            g.pos = (np.asarray(g.pos) + center).tolist()
            g.meshname = ""
            report.replaced.append(label)
        g.margin = cfg.contact_margin
        g.gap = cfg.contact_margin
        report.collidable.append(label)
    if cfg.pads == "capsules":
        _pads_to_capsules(spec, ref, report)
    if cfg.round_objects:
        _round_objects(spec, ref, report)
    if cfg.pads == "capsules" or cfg.round_objects:
        for g in spec.geoms:
            if (g.contype or g.conaffinity):
                g.margin = cfg.contact_margin
                g.gap = cfg.contact_margin
    m = spec.compile()
    # Meta-World resets the mocap weld so that the hand tracks the mocap body.
    for i in range(m.neq):
        if m.eq_type[i] == mujoco.mjtEq.mjEQ_WELD:
            m.eq_data[i] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 5.0]
    m.opt.timestep = cfg.timestep
    m.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    m.opt.iterations = cfg.solver_iterations
    m.opt.ls_iterations = cfg.ls_iterations
    return m, report


def with_contact_band(m: mujoco.MjModel, band: float) -> mujoco.MjModel:
    """Copy of ``m`` whose collidable geoms use ``margin = gap = band``.

    Used for the backward (soft) model only: contact candidates and their
    signed distances then exist up to ``band`` metres before touching, while
    the executed hard model keeps its own band.
    """
    import copy
    m2 = copy.copy(m)
    for i in range(m2.ngeom):
        if m2.geom_contype[i] or m2.geom_conaffinity[i]:
            m2.geom_margin[i] = band
            m2.geom_gap[i] = band
    return m2
