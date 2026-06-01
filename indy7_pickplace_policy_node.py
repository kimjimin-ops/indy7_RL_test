#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
indy7_pickplace_policy_node.py
==============================
Isaac Lab(rsl_rl/PPO)로 학습한 pick&place 정책(policy.onnx)을 ROS2 Jazzy에서
추론하고, FollowJointTrajectory 액션으로 안전하게 실행하는 노드.

구성:
    [관측 조립]  /joint_states + /cube_pose + TF(base->link6)
        -> 24차원 obs (joint_pos8 | joint_vel8 | ee_pos3 | cube_pos3 | grip2)
    [추론]       onnx (obs[1,24] -> actions[1,8]),  정규화 없음
    [안전]       정책출력 제한 / 관절 범위 / 스텝 변화량(속도) 제한
    [실행]       arm 6관절을 FollowJointTrajectory 액션으로 컨트롤러에 전송
                (trajectory 컨트롤러가 보간하므로 raw 명령보다 안전)

이번 단계 범위:
    - gripper(action[6:8])는 계산만 하고 구동하지 않음.
    - 관측의 gripper 자리는 채워야 정책이 정상 동작하므로,
      joint_states에 그리퍼 관절이 없으면(실물 6관절) default(=상대 0)로 채운다.

!! 반드시 검증할 값 !!
    DEFAULT_JOINT_POS : pick&place 학습에 쓴 INDY7_CFG init_state와 일치해야 함.
                        (joint_pos_rel = 현재각 - 기본각 이므로 틀리면 관측 전체가 어긋남)
    JOINT_LIMITS, BASE_LINK, EE_LINK, 토픽/액션 이름 : 실제 환경에 맞게 교체.

타이밍 주의:
    이 노드는 "한 스텝 보내고 완료를 기다린 뒤 다음 스텝"인 안전 스텝 모드다.
    학습 시 30Hz 연속 제어와는 다르므로 동작이 학습과 100% 같지는 않다.
    "정책 출력이 합리적인가 / 큐브로 향하는가"를 안전하게 보는 검증용으로 적합.
"""

import math
import numpy as np
import onnxruntime as ort

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

import tf2_ros
from tf2_ros import TransformException


# =============================================================================
# CONFIG
# =============================================================================

# --- 모델 ---
ONNX_PATH = "/home/kimjimin/test3/logs/rsl_rl/indy7_reach/2026-06-01_14-14-11/exported/policy.onnx"
OBS_DIM = 24
ACT_DIM = 8

# --- 관절 (관측 joint_pos/vel 순서: arm6 + gripper2 = 8) ---
ARM_JOINTS = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
GRIPPER_JOINTS = ["PrismaticJoint", "PrismaticJoint_finger1_b"]
ALL_JOINTS = ARM_JOINTS + GRIPPER_JOINTS

# --- 기본 관절각 (joint_pos_rel 계산용) : INDY7_CFG로 검증 후 교체 ---
#     아래 arm 값은 이전 reach 코드의 기본자세를 가져온 것 (같은 로봇이면 동일할 가능성).
#     pick&place INDY7_CFG의 init_state와 반드시 대조할 것.
DEFAULT_JOINT_POS = {
    "joint0": 0.0, "joint1": 0.0, "joint2": -1.5708,
    "joint3": 0.0, "joint4": -1.5708, "joint5": 0.0,
    "PrismaticJoint": 0.0, "PrismaticJoint_finger1_b": 0.0,
}

# --- 관절 한계 (안전 클리핑, rad) : indy7 사양으로 교체 ---
JOINT_LIMITS = {
    "joint0": (-3.05, 3.05), "joint1": (-3.05, 3.05), "joint2": (-3.05, 3.05),
    "joint3": (-3.05, 3.05), "joint4": (-3.05, 3.05), "joint5": (-3.75, 3.75),
}

# --- 액션 환산 (ActionsCfg와 일치) ---
ARM_ACTION_SCALE = 0.5          # arm_action scale (env 코드 기준)
ARM_USE_DEFAULT_OFFSET = True   # 목표각 = 기본각 + scale*action

# --- 안전 파라미터 ---
RAW_ACTION_CLIP = 1.0           # 정책 출력 자체 제한 (clip_actions: null 보완)
MAX_STEP_DELTA = 0.15           # rad, 한 trajectory에서 현재각 대비 최대 이동량
EXEC_TIME_SEC = 0.5             # 각 trajectory 실행 시간 (속도 = DELTA/EXEC_TIME)

# --- TCP / 좌표 ---
EE_LINK = "link6"
BASE_LINK = "link0"             # Gazebo TF 확인 결과: 로봇 베이스는 link0 (world의 자식)
TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.27])

# --- 토픽 / 액션 ---
JOINT_STATES_TOPIC = "/joint_states"
CUBE_POSE_TOPIC = "/cube_pose"   # base 프레임 기준 PoseStamped 가정
TRAJ_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"

# --- 제어 루프 ---
LOOP_PERIOD_SEC = 0.1            # 타이머 주기 (is_executing 게이트로 실제 스텝은 더 느림)


def quat_apply(q_xyzw, v):
    """벡터 v를 쿼터니언 q(xyzw)로 회전 (ROS 표준 순서)."""
    x, y, z, w = q_xyzw
    uv = np.cross([x, y, z], v)
    uuv = np.cross([x, y, z], uv)
    return v + 2.0 * (w * uv + uuv)


class Indy7PickPlaceNode(Node):
    def __init__(self):
        super().__init__("indy7_pickplace_policy_node")

        # onnx
        self.policy = ort.InferenceSession(
            ONNX_PATH, providers=["CPUExecutionProvider"]
        )
        self.get_logger().info(f"Loaded policy: {ONNX_PATH}")

        # default pose 배열 (ALL_JOINTS 순서)
        self.default_all = np.array(
            [DEFAULT_JOINT_POS[n] for n in ALL_JOINTS], dtype=np.float32
        )
        self.default_arm = np.array(
            [DEFAULT_JOINT_POS[n] for n in ARM_JOINTS], dtype=np.float32
        )

        # 상태
        self.joint_pos = None
        self.joint_vel = None
        self.cube_pos_base = None
        self.is_executing = False

        # TF (EE pose)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # 구독
        self.create_subscription(JointState, JOINT_STATES_TOPIC, self.cb_joints, 10)
        self.create_subscription(PoseStamped, CUBE_POSE_TOPIC, self.cb_cube, 10)

        # 실행: trajectory 액션 클라이언트
        self.action_client = ActionClient(self, FollowJointTrajectory, TRAJ_ACTION)

        # 루프
        self.timer = self.create_timer(LOOP_PERIOD_SEC, self.control_loop)
        self.get_logger().info("Pick&place policy node ready (safe-step mode).")

    # --------------------------------------------------------- 콜백
    def cb_joints(self, msg: JointState):
        self.joint_pos = dict(zip(msg.name, msg.position))
        self.joint_vel = dict(zip(msg.name, msg.velocity)) if msg.velocity else {}

    def cb_cube(self, msg: PoseStamped):
        p = msg.pose.position
        self.cube_pos_base = np.array([p.x, p.y, p.z], dtype=np.float32)

    # --------------------------------------------------------- 관측
    def get_tcp_base(self):
        """TF base->link6 에서 TCP(base 기준) 위치 계산."""
        try:
            tf = self.tf_buffer.lookup_transform(BASE_LINK, EE_LINK, rclpy.time.Time())
        except TransformException as e:
            self.get_logger().warn(f"TF lookup 실패: {e}", throttle_duration_sec=2.0)
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        ee_pos = np.array([t.x, t.y, t.z], dtype=np.float32)
        ee_quat = np.array([r.x, r.y, r.z, r.w], dtype=np.float32)
        return ee_pos + quat_apply(ee_quat, TCP_LOCAL_OFFSET)

    def joint_rel(self, name):
        """관절 상대각. joint_states에 없으면(실물 그리퍼) default 기준 0으로."""
        if name in self.joint_pos:
            return self.joint_pos[name] - DEFAULT_JOINT_POS[name]
        return 0.0  # 미존재 관절은 상대 0

    def joint_vel_of(self, name):
        return self.joint_vel.get(name, 0.0) if self.joint_vel else 0.0

    def assemble_obs(self):
        """학습 ObservationsCfg 순서: joint_pos8 | joint_vel8 | ee3 | cube3 | grip2"""
        if self.joint_pos is None or self.cube_pos_base is None:
            return None
        # arm 관절은 반드시 있어야 함
        for n in ARM_JOINTS:
            if n not in self.joint_pos:
                self.get_logger().warn(f"arm joint '{n}' 없음", throttle_duration_sec=2.0)
                return None

        joint_pos_rel = np.array([self.joint_rel(n) for n in ALL_JOINTS], dtype=np.float32)
        joint_vel_rel = np.array([self.joint_vel_of(n) for n in ALL_JOINTS], dtype=np.float32)

        tcp = self.get_tcp_base()
        if tcp is None:
            return None

        grip_rel = np.array([self.joint_rel(n) for n in GRIPPER_JOINTS], dtype=np.float32)

        obs = np.concatenate(
            [joint_pos_rel, joint_vel_rel, tcp, self.cube_pos_base, grip_rel]
        ).astype(np.float32)

        if obs.shape[0] != OBS_DIM:
            self.get_logger().error(f"obs dim {obs.shape[0]} != {OBS_DIM}")
            return None
        return obs

    # --------------------------------------------------------- 안전
    def safe_arm_target(self, raw_action):
        """정책 arm 출력 6개 -> 안전한 목표각 6개."""
        a = np.clip(raw_action[:6], -RAW_ACTION_CLIP, RAW_ACTION_CLIP)

        if ARM_USE_DEFAULT_OFFSET:
            target = self.default_arm + ARM_ACTION_SCALE * a
        else:
            target = ARM_ACTION_SCALE * a

        # 관절 범위
        lo = np.array([JOINT_LIMITS[n][0] for n in ARM_JOINTS])
        hi = np.array([JOINT_LIMITS[n][1] for n in ARM_JOINTS])
        target = np.clip(target, lo, hi)

        # 현재 측정각 대비 스텝 변화량 제한 (속도 안전)
        cur = np.array([self.joint_pos[n] for n in ARM_JOINTS], dtype=np.float32)
        target = np.clip(target, cur - MAX_STEP_DELTA, cur + MAX_STEP_DELTA)
        return target

    # --------------------------------------------------------- 루프
    def control_loop(self):
        if self.is_executing:
            return
        obs = self.assemble_obs()
        if obs is None:
            return
        if not self.action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn("trajectory 액션 서버 대기 중...", throttle_duration_sec=2.0)
            return

        action = self.policy.run(["actions"], {"obs": obs.reshape(1, OBS_DIM)})[0].reshape(-1)
        arm_target = self.safe_arm_target(action)

        self.get_logger().info(
            "arm_target: " + ", ".join(f"{p:.3f}" for p in arm_target)
        )

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = arm_target.tolist()
        sec = int(EXEC_TIME_SEC)
        nsec = int((EXEC_TIME_SEC - sec) * 1e9)
        point.time_from_start = Duration(sec=sec, nanosec=nsec)
        goal.trajectory.points = [point]

        self.is_executing = True
        fut = self.action_client.send_goal_async(goal)
        fut.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        gh = future.result()
        if not gh.accepted:
            self.get_logger().warn("Goal 거부됨")
            self.is_executing = False
            return
        gh.get_result_async().add_done_callback(self.on_result)

    def on_result(self, future):
        self.is_executing = False


def main():
    rclpy.init()
    node = Indy7PickPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
