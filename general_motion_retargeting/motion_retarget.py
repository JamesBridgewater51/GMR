
import mink
import mujoco as mj
import numpy as np
import json
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from .params import ROBOT_XML_DICT, IK_CONFIG_DICT
from rich import print

class GeneralMotionRetargeting:
    """General Motion Retargeting (GMR).
    """
    def __init__(
        self,
        src_human: str,
        tgt_robot: str,
        actual_human_height: float = None,
        solver: str="daqp", # change from "quadprog" to "daqp".
        damping: float=5e-1, # change from 1e-1 to 1e-2.
        verbose: bool=True,
        use_velocity_limit: bool=False,
    ) -> None:

        # load the robot model
        self.tgt_robot = tgt_robot
        self.xml_file = str(ROBOT_XML_DICT[tgt_robot])
        if verbose:
            print("Use robot model: ", self.xml_file)
        self.model = mj.MjModel.from_xml_path(self.xml_file)
        
        # Print DoF names in order
        if verbose:
            print("[GMR] Robot Degrees of Freedom (DoF) names and their order:")
        self.robot_dof_names = {}
        for i in range(self.model.nv):  # 'nv' is the number of DoFs
            dof_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, self.model.dof_jntid[i])
            self.robot_dof_names[dof_name] = i
            if verbose:
                print(f"DoF {i}: {dof_name}")
            
            
        if verbose:
            print("[GMR] Robot Body names and their IDs:")
        self.robot_body_names = {}
        for i in range(self.model.nbody):  # 'nbody' is the number of bodies
            body_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_BODY, i)
            self.robot_body_names[body_name] = i
            if verbose:
                print(f"Body ID {i}: {body_name}")
        
        if verbose:
            print("[GMR] Robot Motor (Actuator) names and their IDs:")
        self.robot_motor_names = {}
        for i in range(self.model.nu):  # 'nu' is the number of actuators (motors)
            motor_name = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            self.robot_motor_names[motor_name] = i
            if verbose:
                print(f"Motor ID {i}: {motor_name}")

        # Load the IK config
        with open(IK_CONFIG_DICT[src_human][tgt_robot]) as f:
            ik_config = json.load(f)
        if verbose:
            print("Use IK config: ", IK_CONFIG_DICT[src_human][tgt_robot])
        
        # compute the scale ratio based on given human height and the assumption in the IK config
        if actual_human_height is not None:
            ratio = actual_human_height / ik_config["human_height_assumption"]
        else:
            ratio = 1.0
            
        # adjust the human scale table
        for key in ik_config["human_scale_table"].keys():
            ik_config["human_scale_table"][key] = ik_config["human_scale_table"][key] * ratio
    

        # used for retargeting
        self.ik_match_table1 = ik_config["ik_match_table1"]
        self.ik_match_table2 = ik_config["ik_match_table2"]
        self.human_root_name = ik_config["human_root_name"]
        self.robot_root_name = ik_config["robot_root_name"]
        self.use_ik_match_table1 = ik_config["use_ik_match_table1"]
        self.use_ik_match_table2 = ik_config["use_ik_match_table2"]
        self.human_scale_table = ik_config["human_scale_table"]
        self.ground = ik_config["ground_height"] * np.array([0, 0, 1])

        self.max_iter = 10

        self.solver = solver
        self.damping = damping

        self.human_body_to_task1 = {}
        self.human_body_to_task2 = {}
        self.pos_offsets1 = {}
        self.rot_offsets1 = {}
        self.pos_offsets2 = {}
        self.rot_offsets2 = {}

        self.task_errors1 = {}
        self.task_errors2 = {}

        self.ik_limits = [mink.ConfigurationLimit(self.model)]
        if use_velocity_limit:
            # mink.VelocityLimit wants JOINT names; actuator names may carry a
            # robot-specific prefix (e.g. the X2's "motor_*").
            hinge_joints = [
                self.model.joint(j).name for j in range(self.model.njnt)
                if self.model.jnt_type[j] == mj.mjtJoint.mjJNT_HINGE
            ]
            VELOCITY_LIMITS = {k: 3*np.pi for k in hinge_joints}
            self.ik_limits.append(mink.VelocityLimit(self.model, VELOCITY_LIMITS))
            
        self.setup_retarget_configuration()
        
        self.ground_offset = 0.0

    def setup_retarget_configuration(self):
        self.configuration = mink.Configuration(self.model)

        self.x2_arm_geometry = {}
        self.x2_arm_collision_geoms = {}
        self.x2_arm_joint_addrs = {}
        self.x2_arm_joint_ranges = {}
        if self.tgt_robot == "agibot_x2":
            for side in ("left", "right"):
                body_ids = {
                    part: mj.mj_name2id(
                        self.model,
                        mj.mjtObj.mjOBJ_BODY,
                        f"{side}_{part}_link",
                    )
                    for part in ("shoulder_pitch", "shoulder_yaw", "elbow", "wrist_roll")
                }
                xpos = self.configuration.data.xpos
                # Rest OFFSET VECTORS, not scalar lengths: the X2 link origins
                # are not collinear with the anatomical arm axis (at qpos0 the
                # shoulder->elbow origin segment is ~26 deg off vertical), so
                # placing targets at "anchor + direction * length" demands an
                # unnatural pose and the IK answers with a permanent elbow
                # bend.  Rotating the qpos0 offset vectors with the human bone
                # directions keeps the natural configuration reachable: a
                # hanging human arm maps exactly onto the robot's own qpos0.
                self.x2_arm_geometry[side] = {
                    "body_ids": body_ids,
                    # Anchor at shoulder_pitch: its origin is attached to
                    # the torso and does not move with any shoulder joint,
                    # unlike shoulder_yaw whose origin shifts as the arm
                    # moves (rest offsets measured at qpos0 would otherwise
                    # be applied from a moved anchor).
                    "upper_rest": (
                        xpos[body_ids["elbow"]] - xpos[body_ids["shoulder_pitch"]]
                    ).copy(),
                    "forearm_rest": (
                        xpos[body_ids["wrist_roll"]] - xpos[body_ids["elbow"]]
                    ).copy(),
                }
                collision_bodies = {
                    mj.mj_name2id(
                        self.model,
                        mj.mjtObj.mjOBJ_BODY,
                        f"{side}_{part}_link",
                    )
                    for part in ("elbow", "wrist_yaw", "wrist_pitch", "wrist_roll")
                }
                self.x2_arm_collision_geoms[side] = {
                    geom for geom in range(self.model.ngeom)
                    if self.model.geom_bodyid[geom] in collision_bodies
                    and self.model.geom_contype[geom] != 0
                }
                self.x2_arm_joint_addrs[side] = [
                    self.model.joint(name).qposadr[0]
                    for name in (
                        self.model.joint(j).name for j in range(self.model.njnt)
                    )
                    if name.startswith(
                        (f"{side}_shoulder", f"{side}_elbow", f"{side}_wrist")
                    )
                ]
                self.x2_arm_joint_ranges[side] = [
                    (self.model.joint(name).qposadr[0],
                     self.model.joint(name).range[0],
                     self.model.joint(name).range[1])
                    for name in (
                        self.model.joint(j).name for j in range(self.model.njnt)
                    )
                    if name.startswith(
                        (f"{side}_shoulder", f"{side}_elbow")
                    )
                ]
        self.x2_arm_last_targets = {}
        self.x2_arm_solve_data = (
            mj.MjData(self.model) if self.tgt_robot == "agibot_x2" else None
        )
        self._x2_setup_analytic_arms()
        self.x2_foot_tilt_bias = {"left": None, "right": None}
        self.x2_foot_prev_pos = {"left": None, "right": None}
        self.x2_foot_bias_frozen = False
    
        self.tasks1 = []
        self.tasks2 = []
        
        for frame_name, entry in self.ik_match_table1.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task1[body_name] = task
                self.pos_offsets1[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets1[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks1.append(task)
                self.task_errors1[task] = []
        
        for frame_name, entry in self.ik_match_table2.items():
            body_name, pos_weight, rot_weight, pos_offset, rot_offset = entry
            if pos_weight != 0 or rot_weight != 0:
                task = mink.FrameTask(
                    frame_name=frame_name,
                    frame_type="body",
                    position_cost=pos_weight,
                    orientation_cost=rot_weight,
                    lm_damping=1,
                )
                self.human_body_to_task2[body_name] = task
                self.pos_offsets2[body_name] = np.array(pos_offset) - self.ground
                self.rot_offsets2[body_name] = R.from_quat(
                    rot_offset, scalar_first=True
                )
                self.tasks2.append(task)
                self.task_errors2[task] = []

        # Analytic arm mode owns the arm joints entirely: the QP arm tasks
        # are silenced while it is active (and restored for e.g. BVH sources
        # that lack SMPL-X joint rotations).
        self.x2_arm_tasks = []
        self.x2_arm_tasks_disabled = False
        if self.tgt_robot == "agibot_x2":
            for table, mapping in (
                (self.ik_match_table1, self.human_body_to_task1),
                (self.ik_match_table2, self.human_body_to_task2),
            ):
                for entry in table.values():
                    body_name, pos_weight, rot_weight = entry[0], entry[1], entry[2]
                    task = mapping.get(body_name)
                    if task is not None and body_name.endswith(
                            ("_shoulder", "_elbow", "_wrist")):
                        self.x2_arm_tasks.append((task, pos_weight, rot_weight))

        # Null-space regularization: robots whose wrist axes share one origin
        # (e.g. AgiBot X2) leave those joints unconstrained by position-only
        # frame tasks, so they drift into joint limits. A weak posture task
        # pins unconstrained joints to the neutral pose without measurably
        # affecting tracked joints.
        posture_cost = np.full(self.model.nv, 0.05)
        posture_cost[:6] = 0.0  # leave the floating base free
        for j in range(self.model.njnt):
            jname = mj.mj_id2name(self.model, mj.mjtObj.mjOBJ_JOINT, j)
            if jname and ("wrist" in jname or "head" in jname):
                posture_cost[self.model.jnt_dofadr[j]] = 0.5
            # At a lateral T-pose the X2 shoulder is at the pitch-roll-yaw
            # Euler singularity (roll ~= +/-90 deg).  End-effector tasks
            # cannot distinguish the natural pitch=yaw=0 solution from a
            # visibly twisted pitch~=yaw branch.  This remains a soft
            # null-space preference: real forward/back arm motion easily
            # overrides it through the much stronger frame tasks.
            if (self.tgt_robot == "agibot_x2" and jname and
                    ("shoulder_pitch_joint" in jname or
                     "shoulder_yaw_joint" in jname)):
                posture_cost[self.model.jnt_dofadr[j]] = 1.0
            # The X2 torso orientation task otherwise drives waist pitch to its
            # +/-0.314 rad hard stop for most SMPL-X clips.  A strong neutral
            # regularizer trades a small, measurable torso-orientation error
            # for a feasible reference instead of training against a clipped
            # joint.  This is X2-specific; wider-waist robots keep the generic
            # posture cost above.
            if (self.tgt_robot == "agibot_x2"
                    and jname == "waist_pitch_joint"):
                posture_cost[self.model.jnt_dofadr[j]] = 20.0
        posture_task = mink.PostureTask(self.model, cost=posture_cost)
        q_neutral = np.zeros(self.model.nq)
        q_neutral[3] = 1.0
        posture_task.set_target(q_neutral)
        self.tasks1.append(posture_task)
        self.tasks2.append(posture_task)

        # Scratch state for the arm-only nonlinear collision projection.  The
        # analytic arm solver bypasses the whole-body QP collision limits, so
        # the final safety pass below moves shoulder joints only; the waist,
        # legs, root and opposite arm remain bit-identical.
        self.x2_projection_data = (
            mj.MjData(self.model) if self.tgt_robot == "agibot_x2" else None
        )
        self.x2_projection_fromto = np.empty(6)

    def update_targets(self, human_data, offset_to_ground=False):
        # scale human data in local frame
        human_data = self.to_numpy(human_data)
        raw_arm_positions = None
        # SMPL-X joint names only; BVH sources (e.g. LAFAN) name joints
        # differently and skip the X2 arm-target reconstruction entirely.
        if self.x2_arm_geometry and all(
            f"{side}_{part}" in human_data
            for side in ("left", "right")
            for part in ("shoulder", "elbow", "wrist")
        ):
            raw_arm_positions = {
                f"{side}_{part}": np.asarray(
                    human_data[f"{side}_{part}"][0], dtype=np.float64
                ).copy()
                for side in ("left", "right")
                for part in ("shoulder", "elbow", "wrist")
            }
            if self.human_root_name in human_data:
                raw_arm_positions["pelvis"] = np.asarray(
                    human_data[self.human_root_name][0], dtype=np.float64
                ).copy()
        analytic_names = ("spine3",
                          "left_shoulder", "left_elbow", "left_wrist",
                          "right_shoulder", "right_elbow", "right_wrist")
        self.x2_arm_raw_quats = None
        if self.x2_arm_analytic and all(n in human_data for n in analytic_names):
            self.x2_arm_raw_quats = {
                n: np.asarray(human_data[n][1], dtype=np.float64).copy()
                for n in analytic_names
            }
            self.x2_arm_raw_positions = raw_arm_positions
        analytic_active = self.x2_arm_raw_quats is not None
        if analytic_active != self.x2_arm_tasks_disabled:
            for task, pos_weight, rot_weight in self.x2_arm_tasks:
                task.set_position_cost(0.0 if analytic_active else pos_weight)
                task.set_orientation_cost(0.0 if analytic_active else rot_weight)
            self.x2_arm_tasks_disabled = analytic_active
        human_data = self.scale_human_data(human_data, self.human_root_name, self.human_scale_table)
        human_data = self.offset_human_data(human_data, self.pos_offsets1, self.rot_offsets1)
        human_data = self.apply_ground_offset(human_data)
        if offset_to_ground:
            human_data = self.offset_human_data_to_ground(human_data)
        self.scaled_human_data = human_data
        arm_position_targets = {} if analytic_active else (
            self._x2_arm_position_targets(raw_arm_positions, human_data)
        )
        self.x2_arm_last_targets = arm_position_targets
        foot_orientation_targets = self._x2_foot_orientation_targets(human_data)

        if self.use_ik_match_table1:
            for body_name in self.human_body_to_task1.keys():
                task = self.human_body_to_task1[body_name]
                pos, rot = human_data[body_name]
                pos = arm_position_targets.get(body_name, pos)
                rot = foot_orientation_targets.get(body_name, rot)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))
        
        if self.use_ik_match_table2:
            for body_name in self.human_body_to_task2.keys():
                task = self.human_body_to_task2[body_name]
                pos, rot = human_data[body_name]
                pos = arm_position_targets.get(body_name, pos)
                rot = foot_orientation_targets.get(body_name, rot)
                task.set_target(mink.SE3.from_rotation_and_translation(mink.SO3(rot), pos))

    def _x2_arm_position_targets(self, raw_positions, scaled_human_data):
        """Map human arm directions onto X2-reachable elbow/wrist targets.

        Human-scaled arm targets are often beyond the X2's reach; feeding them
        to IK makes the solver roll/yaw the waist toward the extended arm to
        gain reach (boxing clips: waist roll swung +/-27 deg, correlation with
        the punching side +0.63).  Reconstructing targets from human bone
        directions and X2 bone lengths keeps arm extension out of the torso.

        The one exception is a straight arm raised laterally: the X2 shoulder
        is at its Euler singularity there and only the calibrated targets
        select the untwisted branch, so proximity to that pose blends the
        targets back to calibration.  Without a pelvis reference (direct unit
        tests) the lateral test is unavailable and the old bend-only gate is
        used.
        """
        if raw_positions is None:
            return {}

        def smoothstep(x):
            x = np.clip(x, 0.0, 1.0)
            return x * x * (3.0 - 2.0 * x)

        pelvis = raw_positions.get("pelvis")
        torso_up = None
        outward_right = None
        if pelvis is not None:
            shoulders_mid = 0.5 * (
                raw_positions["left_shoulder"] + raw_positions["right_shoulder"]
            )
            up = shoulders_mid - pelvis
            up_norm = np.linalg.norm(up)
            axis = raw_positions["right_shoulder"] - raw_positions["left_shoulder"]
            if up_norm > 1e-8:
                up /= up_norm
                axis = axis - np.dot(axis, up) * up
                axis_norm = np.linalg.norm(axis)
                if axis_norm > 1e-8:
                    torso_up = up
                    outward_right = axis / axis_norm

        targets = {}
        for side, geometry in self.x2_arm_geometry.items():
            shoulder = raw_positions[f"{side}_shoulder"]
            elbow = raw_positions[f"{side}_elbow"]
            wrist = raw_positions[f"{side}_wrist"]
            upper = elbow - shoulder
            forearm = wrist - elbow
            upper_norm = np.linalg.norm(upper)
            forearm_norm = np.linalg.norm(forearm)
            if upper_norm < 1e-8 or forearm_norm < 1e-8:
                continue
            upper /= upper_norm
            forearm /= forearm_norm

            bend = np.degrees(
                np.arccos(np.clip(np.dot(upper, forearm), -1.0, 1.0))
            )
            if outward_right is None:
                # Bend-only fallback: reconstruct only clearly bent arms.
                blend = smoothstep((bend - 20.0) / 30.0)
            else:
                # Reconstruct everywhere except near the lateral straight-arm
                # singularity: straightness and lateralness must both be high
                # for the calibrated targets to take over.
                straightness = smoothstep((50.0 - bend) / 25.0)
                outward = outward_right if side == "right" else -outward_right
                whole = wrist - shoulder
                whole_norm = np.linalg.norm(whole)
                if whole_norm < 1e-8:
                    continue
                lateral_deg = np.degrees(np.arccos(np.clip(
                    np.dot(whole / whole_norm, outward), -1.0, 1.0
                )))
                lateralness = smoothstep((45.0 - lateral_deg) / 25.0)
                blend = 1.0 - straightness * lateralness

            if blend <= 0.0:
                continue

            body_ids = geometry["body_ids"]
            xpos_now = self.configuration.data.xpos
            anchor = xpos_now[body_ids["shoulder_pitch"]].copy()
            # The shoulder TASK tracks the shoulder_yaw link; its target must
            # stay that link's own position (the anchor is pitch, torso-fixed,
            # used only as the base for the elbow/wrist rest offsets).
            robot_shoulder = xpos_now[body_ids["shoulder_yaw"]].copy()
            # Rest offsets were measured at qpos0 (root identity).  Both the
            # bone directions and the offsets must live in the robot's CURRENT
            # root frame: applying qpos0 world-frame vectors to a yawed robot
            # misplaces the targets around the body by the yaw difference.
            qpos_now = self.configuration.data.qpos
            root_now = R.from_quat(qpos_now[3:7], scalar_first=True)

            def swing(direction):
                # Minimal rotation taking the anatomical rest direction
                # (hanging arm, straight down) to the given bone direction,
                # both expressed in the robot root frame.
                down = np.array([0.0, 0.0, -1.0])
                axis = np.cross(down, direction)
                s = np.linalg.norm(axis)
                c = float(np.dot(down, direction))
                if s < 1e-8:
                    if c > 0.0:
                        return R.identity()
                    return R.from_euler("y", 180.0, degrees=True)
                return R.from_rotvec(axis / s * np.arctan2(s, c))

            upper_local = root_now.inv().apply(upper)
            forearm_local = root_now.inv().apply(forearm)
            robot_elbow = anchor + root_now.apply(
                swing(upper_local).apply(geometry["upper_rest"])
            )
            robot_wrist = robot_elbow + root_now.apply(
                swing(forearm_local).apply(geometry["forearm_rest"])
            )
            reconstructed = {
                f"{side}_shoulder": robot_shoulder,
                f"{side}_elbow": robot_elbow,
                f"{side}_wrist": robot_wrist,
            }
            for name, target in reconstructed.items():
                calibrated = scaled_human_data[name][0]
                targets[name] = calibrated + blend * (target - calibrated)
        return targets

    def _x2_foot_orientation_targets(self, human_data):
        """Remove each subject's constant sole-tilt bias from foot targets.

        Mocap-fitted foot joint frames carry the subject's anatomy (arch
        pitch, splay; 5-20 deg across subjects) even when the mesh sole lies
        flat, which tilts the X2 sole in plain standing.  Height-gated
        leveling cannot fix this: heel-strike/toe-off pitch (30 deg, must be
        kept) lives in the same height band as the bias, and switching the
        orientation target with foot height excites the knee (measured
        22.9 deg/frame vs the human's 14.9 max).  Instead the bias is
        estimated online while a foot is the support (slow EMA, converges in
        ~1 s of stance) and subtracted at all times: a quasi-static
        correction with no time-varying gating, so gait dynamics pass
        through untouched and quiet standing ends up flat.
        """
        if self.tgt_robot != "agibot_x2":
            return {}
        names = {side: f"{side}_foot" for side in ("left", "right")}
        if any(n not in human_data for n in names.values()):
            return {}
        heights = {s: float(human_data[n][0][2]) for s, n in names.items()}
        support_level = min(heights.values())

        targets = {}
        up = np.array([0.0, 0.0, 1.0])
        for side, name in names.items():
            rot = R.from_quat(human_data[name][1], scalar_first=True)
            pos = np.asarray(human_data[name][0], dtype=np.float64)
            prev = self.x2_foot_prev_pos[side]
            self.x2_foot_prev_pos[side] = pos.copy()
            # Sample the bias only while the foot is in settled stance.
            # Vertical velocity is the stance signal: a toe-off/heel-strike
            # foot moves vertically (real gait pitch, must not contaminate
            # the bias), while a planted foot has none -- including on
            # treadmill data, where stance feet translate horizontally.
            stationary = prev is not None and abs(pos[2] - prev[2]) < 0.002
            if (not self.x2_foot_bias_frozen
                    and heights[side] - support_level < 0.015 and stationary):
                # Decompose rot = yaw * tilt; the tilt residual while the
                # sole is on the ground is the subject's anatomical bias.
                matrix = rot.as_matrix()
                forward = matrix[:, 0].copy()
                forward[2] = 0.0
                norm = np.linalg.norm(forward)
                if norm > 1e-6:
                    forward /= norm
                    yaw = R.from_matrix(
                        np.column_stack([forward, np.cross(up, forward), up])
                    )
                    tilt = yaw.inv() * rot
                    bias = self.x2_foot_tilt_bias[side]
                    if bias is None:
                        self.x2_foot_tilt_bias[side] = tilt
                    else:
                        step = (bias.inv() * tilt).as_rotvec() * 0.05
                        self.x2_foot_tilt_bias[side] = bias * R.from_rotvec(step)
            bias = self.x2_foot_tilt_bias[side]
            if bias is None:
                continue
            targets[name] = (rot * bias.inv()).as_quat(scalar_first=True)
        return targets

    def precalibrate_x2_foot_bias(self, frames):
        """Batch path: solve each foot's constant sole-tilt bias from a whole
        clip before retargeting, then freeze it.

        A constant correction provably cannot excite joints; the online EMA
        (kept for streaming) can drift mid-clip when stance samples are
        scarce or contaminated, which itself kicked the knee.  Samples are
        settled-stance frames (foot at support level, no vertical motion);
        the componentwise rotvec median rejects outliers.
        """
        if self.tgt_robot != "agibot_x2" or not frames:
            return
        up = np.array([0.0, 0.0, 1.0])
        samples = {"left": [], "right": []}
        prev_z = {"left": None, "right": None}
        for frame in frames:
            if any(f"{s}_foot" not in frame for s in ("left", "right")):
                return
            heights = {s: float(np.asarray(frame[f"{s}_foot"][0])[2])
                       for s in ("left", "right")}
            support_level = min(heights.values())
            for side in ("left", "right"):
                z = heights[side]
                settled = (prev_z[side] is not None
                           and abs(z - prev_z[side]) < 0.002
                           and z - support_level < 0.015)
                prev_z[side] = z
                if not settled:
                    continue
                quat = np.asarray(frame[f"{side}_foot"][1], dtype=np.float64)
                rot = (R.from_quat(quat, scalar_first=True)
                       * self.rot_offsets1[f"{side}_foot"])
                matrix = rot.as_matrix()
                forward = matrix[:, 0].copy()
                forward[2] = 0.0
                norm = np.linalg.norm(forward)
                if norm < 1e-6:
                    continue
                forward /= norm
                yaw = R.from_matrix(
                    np.column_stack([forward, np.cross(up, forward), up])
                )
                samples[side].append((yaw.inv() * rot).as_rotvec())
        for side in ("left", "right"):
            if len(samples[side]) >= 5:
                median = np.median(np.stack(samples[side]), axis=0)
                self.x2_foot_tilt_bias[side] = R.from_rotvec(median)
        self.x2_foot_bias_frozen = True

    def retarget(self, human_data, offset_to_ground=False):
        # Update the task targets
        self.update_targets(human_data, offset_to_ground)
        self._run_ik_stages()
        self._x2_solve_arms_analytic()
        self._x2_recover_stuck_arms()
        self._x2_project_collisions()
        return self.configuration.data.qpos.copy()

    def _x2_project_collisions(self):
        collision_limits = [
            limit for limit in self.ik_limits
            if type(limit).__name__ == "CollisionAvoidanceLimit"
        ]
        if (self.x2_projection_data is None
                or self.x2_arm_raw_quats is None
                or not collision_limits):
            return

        # Work from the concrete geom pairs already configured by the caller,
        # preserving its collision policy.  Add 3 mm to the nominal 5 mm
        # margin so the live joint-velocity clamp cannot immediately put a
        # barely-clear wrist back into the hip shell.
        pairs_by_side = {"left": [], "right": []}
        for limit in collision_limits:
            required = float(limit.minimum_distance_from_collisions) + 0.003
            for geom1, geom2 in limit.geom_id_pairs:
                for side, arm_geoms in self.x2_arm_collision_geoms.items():
                    if geom1 in arm_geoms or geom2 in arm_geoms:
                        pairs_by_side[side].append((geom1, geom2, required))

        qpos = self.configuration.data.qpos.copy()
        scratch = self.x2_projection_data
        fromto = self.x2_projection_fromto

        def separations(side, candidate, addrs):
            scratch.qpos[:] = qpos
            scratch.qpos[addrs] = candidate
            mj.mj_forward(self.model, scratch)
            return np.asarray([
                mj.mj_geomDistance(
                    self.model, scratch, geom1, geom2,
                    max(0.02, required + 0.01), fromto,
                ) - required
                for geom1, geom2, required in pairs_by_side[side]
            ])

        for side, outward in (("left", 1.0), ("right", -1.0)):
            if not pairs_by_side[side]:
                continue
            cal = self.x2_arm_analytic[side]
            shoulder_addrs = cal["addrs"][:3]
            desired = qpos[shoulder_addrs].copy()
            initial = separations(side, desired, shoulder_addrs)
            if initial.size == 0 or np.min(initial) >= 0.0:
                continue

            # Hip contacts are resolved by the smallest outward shoulder-roll
            # change.  A 2 degree bracket plus bisection is deterministic,
            # fast enough for live use, and continuous at the clearance
            # boundary (unlike the old >5 mm penetration on/off switch).
            roll_low, roll_high = cal["ranges"][1]
            max_delta = (roll_high - desired[1] if outward > 0.0
                         else desired[1] - roll_low)
            step = np.radians(2.0)
            bracket_low = 0.0
            bracket_high = None
            best_delta = 0.0
            best_clearance = float(np.min(initial))
            delta = min(step, max_delta)
            while delta > 0.0 and delta <= max_delta + 1e-12:
                candidate = desired.copy()
                candidate[1] += outward * delta
                clearance = float(np.min(
                    separations(side, candidate, shoulder_addrs)
                ))
                if clearance > best_clearance:
                    best_clearance, best_delta = clearance, delta
                if clearance >= 0.0:
                    bracket_high = delta
                    break
                if delta >= max_delta:
                    break
                bracket_low = delta
                delta = min(delta + step, max_delta)

            if bracket_high is not None:
                for _ in range(6):
                    delta = 0.5 * (bracket_low + bracket_high)
                    candidate = desired.copy()
                    candidate[1] += outward * delta
                    if np.min(separations(
                            side, candidate, shoulder_addrs)) >= 0.0:
                        bracket_high = delta
                    else:
                        bracket_low = delta
                corrected = desired.copy()
                corrected[1] += outward * bracket_high
            else:
                # Unusual poses may not clear with pure abduction.  Optimize
                # the three shoulder joints only as a bounded fallback; this
                # still cannot move the waist, root, legs, elbow or wrists.
                seed = desired.copy()
                seed[1] += outward * best_delta
                lower = np.asarray([r[0] for r in cal["ranges"][:3]])
                upper = np.asarray([r[1] for r in cal["ranges"][:3]])

                def residual(candidate):
                    collision = np.maximum(
                        0.0, -separations(side, candidate, shoulder_addrs)
                    )
                    return np.concatenate((candidate - desired,
                                           100.0 * collision))

                result = least_squares(
                    residual, seed, bounds=(lower, upper), max_nfev=25,
                    ftol=1e-7, xtol=1e-7, gtol=1e-7,
                )
                corrected = result.x
            qpos[shoulder_addrs] = corrected

        self.configuration.update(qpos)

    def _run_ik_stages(self):
        if self.use_ik_match_table1:
            # Solve the IK problem
            curr_error = self.error1()
            dt = self.configuration.model.opt.timestep
            vel1 = mink.solve_ik(
                self.configuration, self.tasks1, dt, self.solver, self.damping, limits=self.ik_limits
            )
            self.configuration.integrate_inplace(vel1, dt)
            next_error = self.error1()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                dt = self.configuration.model.opt.timestep
                vel1 = mink.solve_ik(
                    self.configuration, self.tasks1, dt, self.solver, self.damping, limits=self.ik_limits
                )
                self.configuration.integrate_inplace(vel1, dt)
                next_error = self.error1()
                num_iter += 1

        if self.use_ik_match_table2:
            curr_error = self.error2()
            dt = self.configuration.model.opt.timestep
            vel2 = mink.solve_ik(
                self.configuration, self.tasks2, dt, self.solver, self.damping, limits=self.ik_limits
            )
            self.configuration.integrate_inplace(vel2, dt)
            next_error = self.error2()
            num_iter = 0
            while curr_error - next_error > 0.001 and num_iter < self.max_iter:
                curr_error = next_error
                # Solve the IK problem with the second task
                dt = self.configuration.model.opt.timestep
                vel2 = mink.solve_ik(
                    self.configuration, self.tasks2, dt, self.solver, self.damping, limits=self.ik_limits
                )
                self.configuration.integrate_inplace(vel2, dt)

                next_error = self.error2()
                num_iter += 1

    def _x2_arm_residuals(self):
        residuals = {}
        for side, geometry in self.x2_arm_geometry.items():
            if f"{side}_elbow" not in self.x2_arm_last_targets:
                continue
            body_ids = geometry["body_ids"]
            xpos = self.configuration.data.xpos
            residuals[side] = max(
                np.linalg.norm(
                    xpos[body_ids["elbow"]]
                    - self.x2_arm_last_targets[f"{side}_elbow"]
                ),
                np.linalg.norm(
                    xpos[body_ids["wrist_roll"]]
                    - self.x2_arm_last_targets[f"{side}_wrist"]
                ),
            )
        return residuals

    def _x2_recover_stuck_arms(self, threshold=0.10):
        """Escape wrong-branch IK solutions for the arms.

        The reconstructed elbow/wrist targets are reachable by construction,
        so a large steady-state residual means the warm-started QP converged
        into the wrong solution branch (fast motion, e.g. punches, can drag
        the shoulder chain past a singularity; the bad branch then persists
        indefinitely because each frame re-warms from the last).  Recovery is
        purely solver-level: re-seed the offending arm chain at its qpos0
        rest configuration, re-solve the SAME frame's targets, and keep
        whichever solution tracks the targets better.  No motion filtering,
        no posture prior: on healthy frames (take1: residual p90 < 10 cm vs
        20-40 cm when stuck) this never triggers, and a failed recovery
        restores the original solution bit-for-bit.

        A second trigger catches wrapped branches that hide below the
        distance threshold: a shoulder/elbow joint pinned at its hard limit
        while the residual sits well above the healthy median (take1 walking:
        right shoulder pitch pinned at +116.9 deg with 7.6 cm residual vs
        2.6 cm healthy, elbow forced 37 deg past the reference).  A pinned
        limit with leftover task error means the QP is pressing against the
        boundary of the wrong basin; attempting a re-seed is safe because a
        failed attempt is discarded.
        """
        before = self._x2_arm_residuals()
        stuck = [s for s, r in before.items() if r > threshold]
        qpos_now = self.configuration.data.qpos
        for side, residual in before.items():
            if side in stuck or residual <= 0.05:
                continue
            pinned = any(
                qpos_now[addr] - low < 0.02 or high - qpos_now[addr] < 0.02
                for addr, low, high in self.x2_arm_joint_ranges[side]
            )
            if pinned:
                stuck.append(side)
        if not stuck:
            return
        snapshot = self.configuration.data.qpos.copy()
        reseeded = snapshot.copy()
        for side in stuck:
            addrs = self.x2_arm_joint_addrs[side]
            reseeded[addrs] = self.model.qpos0[addrs]
        self.configuration.update(reseeded)
        self._run_ik_stages()
        after = self._x2_arm_residuals()
        if max(after[s] for s in stuck) < max(before[s] for s in stuck) - 0.01:
            return
        self.configuration.update(snapshot)

    def _x2_setup_analytic_arms(self):
        """Calibrate the X2 arm solver.

        Shoulder and elbow follow visible SMPL bone geometry.  Wrist
        orientation still uses the torso-relative SMPL joint rotation, with a
        physical T-pose calibration between the two kinematic conventions.

        Calibration pairs the robot's physical T pose (shoulder roll +/-90,
        every other arm joint 0 -- the untwisted branch asserted by the
        t-pose unit test) with the canonical human T pose (every joint world
        rotation RX90, person facing -Y while the robot model at identity
        faces +X, hence the Rz(90) world alignment inside the torso bridge).
        """
        self.x2_arm_analytic = {}
        self.x2_arm_raw_quats = None
        self.x2_shoulder_max_step = np.radians(35.0)
        if self.tgt_robot != "agibot_x2":
            return
        data_t = mj.MjData(self.model)
        data_t.qpos[:] = self.model.qpos0
        for side, roll_sign in (("left", 1.0), ("right", -1.0)):
            addr = self.model.joint(f"{side}_shoulder_roll_joint").qposadr[0]
            data_t.qpos[addr] = roll_sign * np.pi / 2
        mj.mj_forward(self.model, data_t)

        def body_rot(data, name):
            body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, name)
            return R.from_matrix(data.xmat[body_id].reshape(3, 3))

        pitch_id = mj.mj_name2id(
            self.model, mj.mjtObj.mjOBJ_BODY, "left_shoulder_pitch_link"
        )
        torso_id = int(self.model.body_parentid[pitch_id])
        self.x2_torso_body_id = torso_id
        torso_name = self.model.body(torso_id).name
        rot_torso_t = body_rot(data_t, torso_name)
        rx90 = R.from_euler("x", 90, degrees=True)
        bridge = rot_torso_t.inv() * R.from_euler("z", 90, degrees=True) * rx90

        for side in ("left", "right"):
            def fixed(child):
                # Constant parent-relative mount rotation from the model.
                body_id = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_BODY, f"{side}_{child}_link"
                )
                return R.from_quat(self.model.body_quat[body_id],
                                   scalar_first=True), body_id

            m_pitch, _ = fixed("shoulder_pitch")
            m_roll, _ = fixed("shoulder_roll")
            m_yaw, _ = fixed("shoulder_yaw")
            m_elbow, _ = fixed("elbow")
            m_wrist_yaw, _ = fixed("wrist_yaw")
            m_wrist_pitch, _ = fixed("wrist_pitch")
            m_wrist_roll, _ = fixed("wrist_roll")

            def t_dir(child, parent):
                child_id = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_BODY, f"{side}_{child}_link")
                parent_id = mj.mj_name2id(
                    self.model, mj.mjtObj.mjOBJ_BODY, f"{side}_{parent}_link")
                v = data_t.xpos[child_id] - data_t.xpos[parent_id]
                return rot_torso_t.inv().apply(v / np.linalg.norm(v))

            self.x2_arm_analytic[side] = {
                # Robot bone directions at the physical T pose, torso frame;
                # paired lazily against the SMPL rest bone directions (which
                # droop asymmetrically: the SMPL zero pose is NOT a flat T)
                # measured from the first analytic frame.
                "t_dir_up": t_dir("elbow", "shoulder_pitch"),
                "t_dir_fo": t_dir("wrist_roll", "elbow"),
                "rest_fix": None,
                "bridge": bridge,
                "q_up": rot_torso_t.inv() * body_rot(data_t, f"{side}_shoulder_yaw_link"),
                "q_wr": rot_torso_t.inv() * body_rot(data_t, f"{side}_wrist_roll_link"),
                "mounts": (m_pitch, m_roll, m_yaw, m_elbow,
                           m_wrist_yaw, m_wrist_pitch, m_wrist_roll),
                "addrs": [self.model.joint(f"{side}_{n}_joint").qposadr[0]
                          for n in ("shoulder_pitch", "shoulder_roll",
                                    "shoulder_yaw", "elbow",
                                    "wrist_yaw", "wrist_pitch", "wrist_roll")],
                "ranges": [tuple(self.model.joint(f"{side}_{n}_joint").range)
                           for n in ("shoulder_pitch", "shoulder_roll",
                                     "shoulder_yaw", "elbow",
                                     "wrist_yaw", "wrist_pitch", "wrist_roll")],
                "prev": None,
            }

    def _x2_solve_arms_analytic(self):
        """Overwrite arm joints with bone-led shoulder/elbow and wrist IK."""
        if not self.x2_arm_analytic or self.x2_arm_raw_quats is None:
            return
        data = self.configuration.data
        rot_torso = R.from_matrix(data.xmat[self.x2_torso_body_id].reshape(3, 3))
        rot_h = {k: R.from_quat(v, scalar_first=True)
                 for k, v in self.x2_arm_raw_quats.items()}
        torso_h_inv = rot_h["spine3"].inv()
        qpos = data.qpos.copy()

        def clamp(angle, joint_range):
            return float(np.clip(angle, joint_range[0], joint_range[1]))

        def minimal_rotation(src, dst):
            axis = np.cross(src, dst)
            norm = np.linalg.norm(axis)
            if norm < 1e-9:
                return R.identity()
            angle = np.arctan2(norm, float(np.dot(src, dst)))
            return R.from_rotvec(axis / norm * angle)

        for side, cal in self.x2_arm_analytic.items():
            bridge = cal["bridge"]
            m_pitch, m_roll, m_yaw, m_elbow, m_wyaw, m_wpitch, m_wroll = cal["mounts"]

            if cal["rest_fix"] is None:
                # The SMPL zero pose droops each arm differently (left upper
                # rest direction is ~19 deg off lateral, right ~9 deg): pair
                # "human relative rotation = identity" with the robot T pose
                # rotated per side onto the measured SMPL rest bone
                # directions.  Measured online from the first frame; the
                # ratio R_joint^-1 * bone is a model constant (<0.5 deg
                # spread over take1).
                positions = self.x2_arm_raw_positions

                def rest_dir(distal, proximal, joint_key):
                    v = positions[f"{side}_{distal}"] - positions[f"{side}_{proximal}"]
                    v = v / np.linalg.norm(v)
                    return rot_h[f"{side}_{joint_key}"].inv().apply(v)

                u_rest = rest_dir("elbow", "shoulder", "shoulder")
                f_rest = rest_dir("wrist", "elbow", "elbow")
                upper_world = (
                    positions[f"{side}_elbow"]
                    - positions[f"{side}_shoulder"]
                )
                forearm_world = (
                    positions[f"{side}_wrist"]
                    - positions[f"{side}_elbow"]
                )
                axis_world = np.cross(upper_world, forearm_world)
                axis_world /= np.linalg.norm(axis_world)
                axis_rest = rot_h[f"{side}_shoulder"].inv().apply(axis_world)

                def rest_triad(primary, secondary):
                    z_axis = secondary - np.dot(secondary, primary) * primary
                    z_axis = z_axis / np.linalg.norm(z_axis)
                    return R.from_matrix(np.column_stack(
                        [primary, np.cross(z_axis, primary), z_axis]
                    ))

                cal["u_rest"], cal["f_rest"] = u_rest, f_rest
                cal["triad_up_rest"] = rest_triad(u_rest, axis_rest)
                cal["triad_fo_rest"] = rest_triad(f_rest, axis_rest)
                cal["axis_prev"] = None
                cal["rest_fix"] = (
                    minimal_rotation(cal["t_dir_up"], bridge.apply(u_rest)),
                    minimal_rotation(cal["t_dir_fo"], bridge.apply(f_rest)),
                )
            c_up, c_fo = cal["rest_fix"]

            def to_robot(human_key, q_const):
                rel = torso_h_inv * rot_h[f"{side}_{human_key}"]
                return rot_torso * bridge * rel * bridge.inv() * q_const

            target_wr = to_robot("wrist", c_fo * cal["q_wr"])

            # Preserve the human elbow bend directly.  The shoulder solve
            # below owns only the three shoulder joints; it cannot trade a
            # wrong elbow angle for a smaller position residual.
            positions = self.x2_arm_raw_positions
            upper = (positions[f"{side}_elbow"]
                     - positions[f"{side}_shoulder"])
            forearm = (positions[f"{side}_wrist"]
                       - positions[f"{side}_elbow"])
            upper /= np.linalg.norm(upper)
            forearm /= np.linalg.norm(forearm)
            human_bend = np.arccos(np.clip(
                np.dot(upper, forearm), -1.0, 1.0
            ))
            elbow = clamp(-human_bend, cal["ranges"][3])

            # Solve from the same visible bone geometry used by Any2Any's
            # position tasks.  The raw fitted shoulder-frame rotation is not
            # used: its unobservable axial twist caused the wrapped branches
            # seen in take_01.  A weak forearm constraint disambiguates bend
            # plane; qpos0 only breaks the remaining equal-error twist tie.
            def shoulder_chain(angles):
                return (rot_torso * m_pitch
                        * R.from_euler("y", angles[0]) * m_roll
                        * R.from_euler("x", angles[1]) * m_yaw
                        * R.from_euler("z", angles[2]))

            joint_lo = np.array([cal["ranges"][k][0] for k in range(3)])
            joint_hi = np.array([cal["ranges"][k][1] for k in range(3)])
            prev = cal["prev"]
            solve_data = self.x2_arm_solve_data
            body_ids = self.x2_arm_geometry[side]["body_ids"]
            shoulder_addrs = cal["addrs"][:3]
            elbow_addr = cal["addrs"][3]
            neutral = np.asarray(
                self.model.qpos0[shoulder_addrs], dtype=np.float64
            )
            target_upper = upper

            def geometry_error(candidate):
                solve_data.qpos[:] = qpos
                solve_data.qpos[shoulder_addrs] = candidate
                solve_data.qpos[elbow_addr] = elbow
                mj.mj_forward(self.model, solve_data)
                robot_upper = (
                    solve_data.xpos[body_ids["elbow"]]
                    - solve_data.xpos[body_ids["shoulder_pitch"]]
                )
                robot_forearm = (
                    solve_data.xpos[body_ids["wrist_roll"]]
                    - solve_data.xpos[body_ids["elbow"]]
                )
                robot_upper /= np.linalg.norm(robot_upper)
                robot_forearm /= np.linalg.norm(robot_forearm)
                return np.concatenate((
                    robot_upper - target_upper,
                    0.10 * np.sin(human_bend) * (robot_forearm - forearm),
                    0.05 * (candidate - neutral),
                ))

            # qpos0 only selects the redundant twist branch; its weight is
            # 20x below the tracked upper-arm direction and does not encode a
            # named human pose.
            seeds = [
                neutral if prev is None else np.clip(prev[:3], joint_lo, joint_hi)
            ]
            solutions = []
            for seed in seeds:
                result = least_squares(
                    geometry_error,
                    np.clip(seed, joint_lo + 1e-9, joint_hi - 1e-9),
                    bounds=(joint_lo, joint_hi),
                    max_nfev=15,
                    ftol=1e-8,
                    xtol=1e-8,
                    gtol=1e-8,
                )
                solutions.append((
                    float(np.linalg.norm(geometry_error(result.x))),
                    result.x,
                ))

            anchor = (neutral if prev is None
                      else np.asarray(prev[:3], dtype=np.float64))
            desired = solutions[0][1]
            if prev is None:
                angles = desired
            else:
                angles = anchor + np.clip(
                    desired - anchor,
                    -self.x2_shoulder_max_step,
                    self.x2_shoulder_max_step,
                )
            shoulder = [float(angle) for angle in angles]
            rot_yaw_link = shoulder_chain(angles)

            rot_elbow_link = rot_yaw_link * m_elbow * R.from_euler("y", elbow)

            # Wrist chain: Rz(yaw)*Ry(pitch)*Rx(roll) with tiny mounts.
            g_wrist = (rot_elbow_link * m_wyaw).inv() * target_wr
            wrist_angles = g_wrist.as_euler("ZYX")
            wrist = [clamp(wrist_angles[0], cal["ranges"][4]),
                     clamp(wrist_angles[1], cal["ranges"][5]),
                     clamp(wrist_angles[2], cal["ranges"][6])]

            solution = np.array(shoulder + [elbow] + wrist)
            cal["prev"] = solution
            qpos[cal["addrs"]] = solution
        self.configuration.update(qpos)


    def error1(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks1]
            )
        )
    
    def error2(self):
        return np.linalg.norm(
            np.concatenate(
                [task.compute_error(self.configuration) for task in self.tasks2]
            )
        )


    def to_numpy(self, human_data):
        for body_name in human_data.keys():
            human_data[body_name] = [np.asarray(human_data[body_name][0]), np.asarray(human_data[body_name][1])]
        return human_data


    def scale_human_data(self, human_data, human_root_name, human_scale_table):
        
        human_data_local = {}
        root_pos, root_quat = human_data[human_root_name]
        
        # scale root
        scaled_root_pos = human_scale_table[human_root_name] * root_pos
        
        # scale other body parts in local frame
        for body_name in human_data.keys():
            if body_name not in human_scale_table:
                continue
            if body_name == human_root_name:
                continue
            else:
                # transform to local frame (only position)
                human_data_local[body_name] = (human_data[body_name][0] - root_pos) * human_scale_table[body_name]
            
        # transform the human data back to the global frame
        human_data_global = {human_root_name: (scaled_root_pos, root_quat)}
        for body_name in human_data_local.keys():
            human_data_global[body_name] = (human_data_local[body_name] + scaled_root_pos, human_data[body_name][1])

        return human_data_global
    
    def offset_human_data(self, human_data, pos_offsets, rot_offsets):
        """the pos offsets are applied in the local frame"""
        offset_human_data = {}
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            # apply rotation offset first
            updated_quat = (R.from_quat(quat, scalar_first=True) * rot_offsets[body_name]).as_quat(scalar_first=True)
            offset_human_data[body_name][1] = updated_quat
            
            local_offset = pos_offsets[body_name]
            # compute the global position offset using the updated rotation
            global_pos_offset = R.from_quat(updated_quat, scalar_first=True).apply(local_offset)
            
            offset_human_data[body_name][0] = pos + global_pos_offset
           
        return offset_human_data
            
    def offset_human_data_to_ground(self, human_data):
        """find the lowest point of the human data and offset the human data to the ground"""
        offset_human_data = {}
        ground_offset = 0.1
        lowest_pos = np.inf

        for body_name in human_data.keys():
            # only consider the foot/Foot
            if "Foot" not in body_name and "foot" not in body_name:
                continue
            pos, quat = human_data[body_name]
            if pos[2] < lowest_pos:
                lowest_pos = pos[2]
                lowest_body_name = body_name
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            offset_human_data[body_name] = [pos, quat]
            offset_human_data[body_name][0] = pos - np.array([0, 0, lowest_pos]) + np.array([0, 0, ground_offset])
        return offset_human_data

    def set_ground_offset(self, ground_offset):
        self.ground_offset = ground_offset

    def apply_ground_offset(self, human_data):
        for body_name in human_data.keys():
            pos, quat = human_data[body_name]
            human_data[body_name][0] = pos - np.array([0, 0, self.ground_offset])
        return human_data
