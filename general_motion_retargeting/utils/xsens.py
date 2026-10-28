import general_motion_retargeting.utils.lafan_vendor.utils as utils
from general_motion_retargeting.utils.xsens_vendor.BVHParser import BVHParser, Anim
import json
import numpy as np
from pathlib import Path


def _load_joint_offsets(joint_names, path="offsets.json"):
    """Load optional BVH Euler offsets without importing the PyQt curve editor."""
    offset_path = Path(path)
    if offset_path.exists():
        try:
            offsets = json.loads(offset_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Unable to read {offset_path}: {exc}; using zero offsets")
            offsets = {}
    else:
        offsets = {}
    axes = ("X", "Y", "Z")
    return np.asarray(
        [[float(offsets.get(joint, {}).get(axis, 0.0)) for axis in axes] for joint in joint_names],
        dtype=np.float64,
    )


def bvh_parse(args):
    parser = BVHParser(axis_order="zxy", scale=args.scale)
    with open(args.bvh_file, "r") as f:
        bvh_text = f.read()
    rotations, positions = parser.parse(
        bvh_text, start=args.start, end=args.end, reset_to_zero=args.reset_to_zero
    )
    joint_offset = _load_joint_offsets(parser.names)
    new_rotations = rotations + joint_offset
    positions = np.copy(parser.positions)
    _quats, _positions, _offsets, _parents = parser._MOTION_data_post_processing(
        new_rotations, positions, reset_to_zero=True
    )
    print("MOTION_data_post_processing")
    anim = Anim(_quats, _positions, _offsets, _parents, parser.names)
    global_data = utils.quat_fk(anim.quats, anim.pos, anim.parents)
    return anim, global_data, parser.frame_time


def load_xsens_file(args):
    """
    Must return a dictionary with the following structure:
    {
        "Hips": (position, orientation),
        "Spine": (position, orientation),
        ...
    }
    """
    anim, global_data, frame_time = bvh_parse(args)
    frames = []
    for frame in range(anim.pos.shape[0]):
        result = {}
        for i, bone in enumerate(anim.bones):
            orientation = global_data[0][frame, i]
            position = global_data[1][frame, i]
            result[bone] = (position, orientation)

        # Add modified foot pose
        # To make the config file more universal,
        # here the descriptions of the key points of the bvh file
        # that xsens may obtain are aligned with Lafan1
        if args.bvh_format == "3DSM":
            result["LeftFootMod"] = (
                np.array(
                    [
                        result["LeftAnkle"][0][0],
                        result["LeftAnkle"][0][1],
                        result["LeftAnkle"][0][2],
                        # result["LeftToe"][0][2],
                    ]
                ),
                result["LeftAnkle"][1],
                # result["LeftToe_end_site"][1],
            )
            result["RightFootMod"] = (
                np.array(
                    [
                        result["RightAnkle"][0][0],
                        result["RightAnkle"][0][1],
                        result["RightAnkle"][0][2],
                        # result["RightToe"][0][2],
                    ]
                ),
                result["RightAnkle"][1],
                # result["RightToe_end_site"][1],
            )

            # result["Spine2"] = result.pop("Chest4")

        frames.append(result)

    human_height = result["Head_end_site"][0][2] - min(
        result["LeftToe_end_site"][0][2], result["RightToe_end_site"][0][2]
    )
    return frames, human_height, frame_time
