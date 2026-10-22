"""Regression test for forwarding limits to Mink."""

import numpy as np
import pytest

import mink
from general_motion_retargeting import GeneralMotionRetargeting

from synthetic_motion import build_frames


def test_solve_ik_receives_limits_keyword(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pass IK limits by keyword without setting safety_break."""
    retargeter = GeneralMotionRetargeting(
        "smplx", "unitree_g1", verbose=False, use_velocity_limit=True
    )
    calls = []

    def recording_solve_ik(*args, **kwargs):
        calls.append((args, kwargs))
        return np.zeros(retargeter.model.nv)

    monkeypatch.setattr(mink, "solve_ik", recording_solve_ik)
    retargeter.retarget(dict(build_frames()[0]))

    assert calls
    for args, kwargs in calls:
        assert len(args) == 5
        assert kwargs["limits"] is retargeter.ik_limits
        assert "safety_break" not in kwargs
