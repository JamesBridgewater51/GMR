"""Synthetic marching motion for golden regression tests."""

import math
from collections.abc import Mapping
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

FPS: int = 30
NUM_FRAMES: int = 90
STRIDE_HZ: float = 1.0
FORWARD_SPEED: float = 0.3

# Upright SMPL-X T-pose converted to GMR's z-up coordinates.
TPOSE_QUAT: tuple[float, float, float, float] = (0.5, 0.5, 0.5, 0.5)

# T-pose body positions for a 1.8 m human, z-up, facing +x, left = +y.
TPOSE_POSITIONS: Mapping[str, tuple[float, float, float]] = {
    "pelvis": (0.0, 0.0, 0.95),
    "spine3": (0.0, 0.0, 1.25),
    "left_hip": (0.0, 0.09, 0.90),
    "right_hip": (0.0, -0.09, 0.90),
    "left_knee": (0.0, 0.10, 0.50),
    "right_knee": (0.0, -0.10, 0.50),
    "left_foot": (0.0, 0.11, 0.08),
    "right_foot": (0.0, -0.11, 0.08),
    "left_shoulder": (0.0, 0.18, 1.40),
    "right_shoulder": (0.0, -0.18, 1.40),
    "left_elbow": (0.0, 0.45, 1.40),
    "right_elbow": (0.0, -0.45, 1.40),
    "left_wrist": (0.0, 0.70, 1.40),
    "right_wrist": (0.0, -0.70, 1.40),
}

Vector: TypeAlias = NDArray[np.float64]
Frame: TypeAlias = Mapping[str, tuple[Vector, Vector]]


def _build_frame(t: float) -> Frame:
    phase = 2.0 * math.pi * STRIDE_HZ * t
    swing = math.sin(phase)
    forward = FORWARD_SPEED * t
    bob = 0.03 * math.sin(2.0 * phase)

    frame: dict[str, tuple[Vector, Vector]] = {}
    for body, (x, y, z) in TPOSE_POSITIONS.items():
        px, py, pz = x + forward, y, z + bob
        if body == "left_foot":
            pz += 0.06 * max(0.0, swing)
            px += 0.08 * swing
        if body == "right_foot":
            pz += 0.06 * max(0.0, -swing)
            px -= 0.08 * swing
        if body in ("left_elbow", "left_wrist"):
            px -= 0.10 * swing
        if body in ("right_elbow", "right_wrist"):
            px += 0.10 * swing
        pos = np.array([round(v, 6) for v in (px, py, pz)])
        frame[body] = (pos, np.array(TPOSE_QUAT))
    return frame


def build_frames() -> tuple[Frame, ...]:
    """Build the synthetic motion frames."""
    return tuple(_build_frame(i / FPS) for i in range(NUM_FRAMES))
