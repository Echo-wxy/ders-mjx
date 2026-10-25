"""Scripted controllers that drive each scene through its contact events.

The controllers only generate evaluation states (free motion, approach,
contact formation, sustained contact, grasping, separation). They are simple
phase machines on the hand and object positions with Gaussian action noise.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx

GAIN = 40.0


def _toward(hand, target):
    return np.clip(GAIN * (target - hand), -1.0, 1.0)


class ScriptedController:
    def __init__(self, scene, rng: np.random.Generator):
        self.scene, self.rng = scene, rng
        self.phase, self.timer = 0, 0
        self.goal = rng.uniform([-0.2, 0.45, 0.08], [0.2, 0.8, 0.3])  # reach target

    def __call__(self, d) -> np.ndarray:
        hand, obj = self.scene.hand_obj(d)
        name = self.scene.spec.name
        grip = -1.0
        self.timer += 1
        if name == "reach":
            move = _toward(hand, self.goal)
        elif name in ("push", "plate_slide"):
            behind = obj + np.array([0.0, -0.06, 0.0])
            low = 0.05
            if self.phase == 0:   # go above the point behind the object
                tgt = np.r_[behind[:2], 0.12]
                if np.linalg.norm(hand[:2] - tgt[:2]) < 0.01:
                    self.phase = 1
            elif self.phase == 1:  # descend
                tgt = np.r_[behind[:2], low]
                if hand[2] < low + 0.01:
                    self.phase = 2
            else:                  # push forward through the object
                tgt = np.r_[obj[:2] + np.array([0.0, 0.10]), low]
            move, grip = _toward(hand, tgt), 1.0
        elif name in ("pick_place", "peg_insert"):
            if self.phase == 0:    # above the object, gripper open
                tgt = np.r_[obj[:2], 0.12]
                if np.linalg.norm(hand[:2] - obj[:2]) < 0.01:
                    self.phase = 1
            elif self.phase == 1:  # descend around the object
                tgt = np.r_[obj[:2], max(obj[2], 0.05)]
                if hand[2] < max(obj[2], 0.05) + 0.01:
                    self.phase, self.timer = 2, 0
            elif self.phase == 2:  # close
                tgt, grip = hand, 1.0
                if self.timer > 15:
                    self.phase = 3
            else:                  # lift and carry
                grip = 1.0
                if name == "pick_place":
                    tgt = np.array([0.0, 0.8, 0.25])
                else:              # toward the hole of the box
                    tgt = self.scene.site_pos(d, "hole") + np.array([0.12, 0.0, 0.0])
            if self.phase != 2:
                grip = -1.0 if self.phase < 2 else 1.0
            move = _toward(hand, tgt)
        else:
            raise ValueError(name)
        a = np.r_[move, grip] + self.rng.normal(0.0, 0.1, 4)
        # Kept strictly inside the action box so that the action Jacobian is
        # not evaluated on the clipping boundary.
        return np.clip(a, -0.95, 0.95)


def sample_states(scene, step_hard, seed: int, n_steps: int = 200):
    """Roll out the scripted controller in the hard engine.

    Returns a list of ``(mjx.Data, action)`` pairs, one per step, where the
    action is the one taken in that state.
    """
    rng = np.random.default_rng(seed)
    d = mjx.put_data(scene.mj, scene.reset_data(rng))
    ctrl = ScriptedController(scene, rng)
    out = []
    for _ in range(n_steps):
        a = jp.asarray(ctrl(d), dtype=d.qpos.dtype)
        out.append((d, a))
        d = step_hard(d, a)
    return out
