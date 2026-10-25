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
