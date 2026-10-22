"""Golden-trajectory regression tests for the retargeting pipeline.

Regenerate the expected trajectories with:

    GMR_REGEN_GOLDENS=1 uv run pytest tests/test_retarget_regression.py
"""

import os
import pathlib
from collections.abc import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from synthetic_motion import Frame, build_frames

from general_motion_retargeting import GeneralMotionRetargeting

DATA_DIR: pathlib.Path = pathlib.Path(__file__).parent / "data"

ROBOTS: tuple[str, ...] = ("unitree_g1", "booster_t1")
REQUIRED_TRACKING_BODIES: tuple[str, ...] = (
    "pelvis",
    "left_foot",
    "right_foot",
)

# Allow small differences between BLAS implementations.
ATOL: float = 1e-4

# Maximum position error in meters for required IK tasks.
TRACKING_ATOL: float = 0.05


def load_synthetic_motion() -> tuple[Frame, ...]:
    """Load the deterministic human motion used by regression tests."""
    return build_frames()


def retarget_motion(robot: str, frames: Sequence[Frame]) -> NDArray[np.float64]:
    """Retarget a sequence of human frames to one robot."""
    retargeter = GeneralMotionRetargeting("smplx", robot, verbose=False)
    return np.stack([retargeter.retarget(dict(frame)) for frame in frames])


@pytest.fixture(scope="module")
def synthetic_frames() -> tuple[Frame, ...]:
    """Provide deterministic human motion frames."""
    return load_synthetic_motion()


@pytest.mark.parametrize("robot", ROBOTS)
def test_qpos_matches_golden(
    robot: str, synthetic_frames: Sequence[Frame]
) -> None:
    qpos = retarget_motion(robot, synthetic_frames)
    golden_path = DATA_DIR / f"golden_qpos_{robot}.npy"
    if os.environ.get("GMR_REGEN_GOLDENS") == "1":
        np.save(golden_path, qpos)
        pytest.skip(f"regenerated {golden_path.name}; verify the diff and commit")
    golden = np.load(golden_path)
    assert qpos.shape == golden.shape
    np.testing.assert_allclose(qpos, golden, rtol=0.0, atol=ATOL)


@pytest.mark.parametrize("robot", ROBOTS)
def test_qpos_is_sane(robot: str, synthetic_frames: Sequence[Frame]) -> None:
    qpos = retarget_motion(robot, synthetic_frames)
    assert np.isfinite(qpos).all()
    # The synthetic subject stands upright; the floating-base height must
    # stay in a plausible standing range for a human-scale robot.
    assert (qpos[:, 2] > 0.3).all()
    assert (qpos[:, 2] < 1.5).all()
    # The root quaternion must stay normalized.
    norms = np.linalg.norm(qpos[:, 3:7], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)


@pytest.mark.parametrize("robot", ROBOTS)
def test_required_tasks_converge(
    robot: str, synthetic_frames: Sequence[Frame]
) -> None:
    """Check that pelvis and foot position tasks converge in both IK stages."""
    retargeter = GeneralMotionRetargeting("smplx", robot, verbose=False)
    task_groups = (
        ("stage1", retargeter.human_body_to_task1),
        ("stage2", retargeter.human_body_to_task2),
    )
    missing = [
        f"{stage}:{body}"
        for stage, task_by_body in task_groups
        for body in REQUIRED_TRACKING_BODIES
        if body not in task_by_body
    ]
    assert not missing, f"{robot} is missing required IK tasks: {missing}"

    tasks = {
        f"{stage}:{body}": task_by_body[body]
        for stage, task_by_body in task_groups
        for body in REQUIRED_TRACKING_BODIES
    }
    worst_errors = dict.fromkeys(tasks, 0.0)
    for frame in synthetic_frames:
        retargeter.retarget(dict(frame))
        for name, task in tasks.items():
            error = task.compute_error(retargeter.configuration)
            worst_errors[name] = max(
                worst_errors[name], float(np.linalg.norm(error[:3]))
            )

    failures = {
        name: error for name, error in worst_errors.items() if error >= TRACKING_ATOL
    }
    assert not failures, (
        f"{robot} required IK tasks did not converge below "
        f"{TRACKING_ATOL} m: {failures}"
    )
