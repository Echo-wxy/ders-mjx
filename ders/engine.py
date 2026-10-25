"""Meta-World scenes on MJX and the transitions compared in the paper.

Every transition executes the same hard MuJoCo/MJX step. They differ only in
how a derivative is produced:

``hard_ad``   automatic differentiation through the hard step itself;
``cfd``       Contacts From Distance of the MJX fork (straight-through);
``ders``      DERS: logistic gate on the signed distance of every contact
              candidate (straight-through);
``generic``   conventional relaxation used only in the backward pass: the
              contact time constant is enlarged (compliant contact), the
              activation stays the hard indicator (straight-through);
``rs``        randomized smoothing (zeroth-order, antithetic Gaussian samples).

Central finite differences of the hard step are the reference and are
provided by :func:`fd_jacobian`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx

from . import soft_contact  # noqa: F401  (installs the gated soft rows)
from .soft_contact import coupling_form, soft_mode
from .metaworld_patch import PatchConfig, load_patched_model

ACTION_SCALE = 0.01
FRAME_SKIP = 5
HAND_INIT = np.array([0.0, 0.6, 0.2])
MOCAP_QUAT = np.array([1.0, 0.0, 1.0, 0.0]) / np.sqrt(2.0)
MOCAP_LOW = np.array([-0.5, 0.40, 0.05])
MOCAP_HIGH = np.array([0.5, 1.0, 0.5])


@dataclass(frozen=True)
class Temperatures:
    """Relaxation parameters of the soft model."""

    alpha_ders: float = 500.0     # DERS contact gate sharpness, 1/m
    soften_factor: float = 10.0   # generic STE: contact time constant multiplied by this factor
    cfd_width: float = 0.01       # CFD influence width, m (equal to the contact margin)


@dataclass(frozen=True)
class SceneSpec:
    """Task scene: Meta-World XML, the manipulated object and its contact partners."""

    name: str
    xml: str
    obj_body: str | None = None     # body name of the object, or
    obj_joint: str | None = None    # name of one of its joints (unnamed bodies)
    env_bodies: tuple = ()          # non-support bodies the object can hit (e.g. a hole)
    obj_xy_low: tuple = (-0.1, 0.6)
    obj_xy_high: tuple = (0.1, 0.7)


SCENES = {
    "reach": SceneSpec("reach", "sawyer_reach_v3.xml", obj_body="obj"),
    "push": SceneSpec("push", "sawyer_push_v3.xml", obj_body="obj"),
    "pick_place": SceneSpec("pick_place", "sawyer_pick_place_v3.xml", obj_body="obj"),
    "plate_slide": SceneSpec("plate_slide", "sawyer_plate_slide.xml", obj_joint="goal_slidex",
                             obj_xy_low=(-0.02, -0.02), obj_xy_high=(0.02, 0.02)),
    "peg_insert": SceneSpec("peg_insert", "sawyer_peg_insertion_side.xml", obj_body="peg",
                            env_bodies=("box",), obj_xy_low=(0.0, 0.5), obj_xy_high=(0.2, 0.7)),
}



class Scene:
    """One Meta-World scene with patched collision geometry."""

    def __init__(self, spec: SceneSpec | str, patch: PatchConfig | None = None):
        if isinstance(spec, str):
            spec = SCENES[spec]
        self.spec = spec
        patch = patch or PatchConfig()
        self.mj, self.report = load_patched_model(spec.xml, patch)
        m = self.mj
        mx = mjx.put_model(m)
        # Options introduced by the MJX fork; set explicitly (they default to None).
        self.mx = mx.replace(opt=mx.opt.replace(
            cfd_enable=False,
            cfd_solimp=jp.array([0.0, 0.01, patch.contact_margin, 1.0, 2.0]),
            col_soft_enable=False, softjax_mode="hard", col_softness=0.1,
            scan_loop=True))  # fixed solver iterations: needed for reverse-mode AD, same for all methods
        self.hand_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.pad_ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                 for b in ("rightpad", "leftpad")])
        if spec.obj_body is not None:
            self.obj_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, spec.obj_body)
        else:
            j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, spec.obj_joint)
            self.obj_id = int(m.jnt_bodyid[j])
        jnts = [j for j in range(m.njnt) if m.jnt_bodyid[j] == self.obj_id]
        self.obj_free = len(jnts) == 1 and m.jnt_type[jnts[0]] == mujoco.mjtJoint.mjJNT_FREE
        self.obj_qadr = [int(m.jnt_qposadr[j]) for j in jnts]
        self.obj_dadr = [int(m.jnt_dofadr[j]) for j in jnts]
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self.gripper_geoms = np.array([i for i, n in enumerate(names)
                                       if n.startswith(("rightpad_geom", "leftpad_geom", "rightclaw_it", "leftclaw_it"))])
        collidable = lambda i: bool(m.geom_contype[i] or m.geom_conaffinity[i])
        self.obj_geoms = np.array([i for i in range(m.ngeom)
                                   if m.geom_bodyid[i] == self.obj_id and collidable(i)])
        env_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in spec.env_bodies]
        self.env_geoms = np.array([i for i in range(m.ngeom)
                                   if m.geom_bodyid[i] in env_ids and collidable(i)], dtype=int)

    # ------------------------------------------------------------------ reset
    def reset_data(self, rng: np.random.Generator, settle_steps: int = 50) -> mujoco.MjData:
        """Meta-World style reset: settle the hand at its initial pose, place the object."""
        m = self.mj
        d = mujoco.MjData(m)
        mujoco.mj_resetData(m, d)
        d.mocap_pos[0] = HAND_INIT
        d.mocap_quat[0] = MOCAP_QUAT
        xy = rng.uniform(self.spec.obj_xy_low, self.spec.obj_xy_high)
        if self.obj_free:
            q = self.obj_qadr[0]
            d.qpos[q:q + 2] = xy
            d.qpos[q + 2] = max(d.qpos[q + 2], 0.02)
            d.qpos[q + 3:q + 7] = [1.0, 0.0, 0.0, 0.0]
        else:  # slide joints: offsets from the default pose
            for k, q in enumerate(self.obj_qadr[:2]):
                d.qpos[q] = xy[k]
        for _ in range(settle_steps * FRAME_SKIP):  # Meta-World: 50 env steps
            d.mocap_pos[0] = HAND_INIT
            d.ctrl[:] = [-1.0, 1.0]
            mujoco.mj_step(m, d)
        return d

    # -------------------------------------------------------------- features
    def features(self, d: mjx.Data) -> jax.Array:
        """12-dim state used for Jacobian comparisons.

        tool-centre-point position (3), object position (3), finger joint positions (2),
        object linear velocity (3), object angular velocity about z (1).
        For an object on two slide joints the last four entries are the two
        slide velocities followed by zeros.
        """
        if self.obj_free:
            v = self.obj_dadr[0]
            vel = jp.concatenate([d.qvel[v:v + 3], d.qvel[v + 5:v + 6]])
        else:
            vel = jp.concatenate([d.qvel[jp.array(self.obj_dadr[:2])], jp.zeros(2, d.qvel.dtype)])
        return jp.concatenate([self.tcp(d), d.xpos[self.obj_id], d.qpos[7:9], vel])

    def tcp(self, d):
        """Tool centre point: midpoint between the two finger pads (Meta-World)."""
        return 0.5 * (d.xpos[self.pad_ids[0]] + d.xpos[self.pad_ids[1]])

    def site_pos(self, d, name: str) -> np.ndarray:
        return np.asarray(d.site_xpos[mujoco.mj_name2id(self.mj, mujoco.mjtObj.mjOBJ_SITE, name)])

    def hand_obj(self, d) -> tuple[np.ndarray, np.ndarray]:
        return np.asarray(self.tcp(d)), np.asarray(d.xpos[self.obj_id])


# --------------------------------------------------------------------- steps
def _apply_action(d: mjx.Data, a: jax.Array) -> mjx.Data:
    """Meta-World action: mocap displacement (clipped to the workspace) and gripper."""
    a = jp.clip(a, -1.0, 1.0)
    pos = jp.clip(d.mocap_pos[0] + ACTION_SCALE * a[:3], MOCAP_LOW, MOCAP_HIGH)
    return d.replace(mocap_pos=pos[None], ctrl=jp.stack([a[3], -a[3]]))


def _substeps(mx: mjx.Model, d: mjx.Data) -> mjx.Data:
    def body(dd, _):
        nxt = mjx.step(mx, dd)
        # keep the carry dtypes fixed (integer fields can be promoted under x64)
        return jax.tree.map(lambda x, y: x.astype(y.dtype), nxt, dd), None
    d, _ = jax.lax.scan(body, d, None, length=FRAME_SKIP)
    return d


METHODS = ("hard_ad", "cfd", "ders", "generic")


def default_params(temps: Temperatures = Temperatures(), dtype=None) -> dict:
    """Relaxation parameters as a pytree of scalars (traced inside jit)."""
    dtype = dtype or jp.zeros(()).dtype
    return {
        "alpha": jp.asarray(temps.alpha_ders, dtype),
        "soften": jp.asarray(temps.soften_factor, dtype),
        "cfd_width": jp.asarray(temps.cfd_width, dtype),
    }


def make_transition(scene: Scene, method: str, temps: Temperatures = Temperatures(),
                    coupling: str = "exact"):
    """Return ``step(d, a, params=None) -> d_next``.

    The forward value is the hard MuJoCo/MJX step for every method; ``method``
    selects the backward model. ``params`` (see :func:`default_params`) holds
    the relaxation parameters as traced scalars, so a temperature sweep reuses
    one compiled program. ``coupling`` selects the straight-through form
    ("exact" or the left-to-right "rounding" form, used only in the
    forward-consistency experiment).
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    base = scene.mx
    defaults = default_params(temps)

    def step(d: mjx.Data, a: jax.Array, params: dict | None = None) -> mjx.Data:
        p = defaults if params is None else params
        opt = base.opt.replace(cfd_solimp=jp.stack([0.0 * p["cfd_width"], 0.01 + 0.0 * p["cfd_width"],
                                                    p["cfd_width"], 1.0 + 0.0 * p["cfd_width"],
                                                    2.0 + 0.0 * p["cfd_width"]]),
                               cfd_enable=method != "hard_ad")
        mx = base.replace(opt=opt)
        if method == "ders":
            ctx = soft_mode("gate", p["alpha"])
        elif method == "generic":
            ctx = soft_mode("soften", p["soften"])
        else:
            ctx = soft_mode("cfd", 0.0)
        with ctx, coupling_form(coupling):
            return _substeps(mx, _apply_action(d, a))

    return step


def hard_step(scene: Scene):
    return make_transition(scene, "hard_ad")
