"""
rl_policy_node2_sim.py  (안전 검증용 — 드라이런 + 시뮬 퍼블리셔)
─────────────────────────────────────────────────────────────
실제 로봇은 움직이지 않고, Isaac Sim의 로봇만 정책 명령대로 동작시켜
안전하게 검증하기 위한 버전.

[동작 구조]
  - obs용 /joint_states 구독: 실제 로봇의 현재 자세를 읽음 (정책 입력)
  - /cube_position 구독: 카메라가 검출한 큐브 위치
  - DRY_RUN=True 이면 실제 로봇으로 FollowJointTrajectory를 보내지 않음
  - 대신 arm_target을 /sim_joint_command (JointState)로 퍼블리시 → Isaac Sim 로봇이 따라감

[USDA 설정 필요]
  ActionGraph의 ros2_subscribe_joint_state 노드 topicName:
      "joint_states"  →  "sim_joint_command"  로 변경할 것
  (안 바꾸면 시뮬이 실제 로봇 상태 토픽을 명령으로 오해함)

[안전장치]
  1. cube_pos z = 0.035 고정 (학습 분포와 일치)
  2. 관절 한계 클램핑 (joint0~4: ±175°, joint5: ±215°)
  3. 직전 명령 대비 변화량 제한 (MAX_DELTA_RAD)
  4. DRY_RUN 플래그로 실로봇 전송 on/off

[검증이 끝난 뒤]
  DRY_RUN = False 로 바꾸면 실제 로봇으로 명령 전송 시작.
"""
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PointStamped
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from tf2_ros import Buffer, TransformListener

import onnxruntime as ort
import numpy as np
import math


# ============================================================
# ★★★ 안전 검증 스위치 ★★★
#   True  : 실제 로봇으로 명령 안 보냄 (시뮬만 동작) — 검증용
#   False : 실제 로봇으로 명령 전송      — 검증 끝난 뒤에만!
# ============================================================
DRY_RUN = True

# 시뮬레이터로 명령을 보낼 토픽 (USDA subscribe 노드와 일치시킬 것)
SIM_COMMAND_TOPIC = '/sim_joint_command'

# ============================================================
# default joint positions (INDY7_CFG init_state 와 일치)
# ============================================================
DEFAULT_ARM_POS = np.array(
    [0.0, 0.0, -1.5708, 0.0, -1.5708, 0.0], dtype=np.float32
)
DEFAULT_GRIPPER_POS = np.array([0.015, 0.015], dtype=np.float32)
DEFAULT_JOINT_POS = np.concatenate([DEFAULT_ARM_POS, DEFAULT_GRIPPER_POS])

# env_cfg ActionsCfg 와 일치
ARM_SCALE = 0.5

# link6 → TCP 오프셋 (env_cfg ee_position_w 와 동일)
TCP_LOCAL_OFFSET = np.array([0.0, 0.0, 0.27])

# 큐브 z 고정값 (학습 분포: cube 월드 0.925 - base 0.89 = 0.035)
CUBE_Z_FIXED = 0.035

# 관절 이름
ARM_JOINT_NAMES = ['joint0', 'joint1', 'joint2', 'joint3', 'joint4', 'joint5']

# ============================================================
# 안전장치 파라미터
# ============================================================
# Indy7 관절 한계 (USDA physics limit: j0~4 ±175°, j5 ±215°)
JOINT_LIMITS_LOW = np.radians(
    np.array([-175, -175, -175, -175, -175, -215], dtype=np.float32)
)
JOINT_LIMITS_HIGH = np.radians(
    np.array([175, 175, 175, 175, 175, 215], dtype=np.float32)
)

# 한 스텝에 허용하는 최대 관절 변화량 (라디안). 0.26rad ≈ 15°
MAX_DELTA_RAD = math.radians(15.0)


def quat_apply(q, v):
    """쿼터니언 q=[w,x,y,z]로 벡터 v 회전 (numpy 단일 벡터)."""
    w, x, y, z = q
    t = 2.0 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


class RLPolicyNodeSim(Node):
    def __init__(self):
        super().__init__('rl_policy_node_sim')

        # ── ONNX 정책 로드 ──
        onnx_path = (
            '/home/kimjimin/test3/logs/rsl_rl/indy7_reach/'
            '2026-06-18_15-23-13/exported/policy.onnx'
        )
        self.policy = ort.InferenceSession(onnx_path)
        self.get_logger().info(f'ONNX 정책 로드: {onnx_path}')

        mode = 'DRY-RUN (시뮬만)' if DRY_RUN else '★ 실제 로봇 전송 ★'
        self.get_logger().info(f'모드: {mode}')

        # ── 상태 변수 ──
        self.latest_joint_state = None
        self.latest_cube_pos = None
        self.is_executing = False
        self.prev_target = None      # 직전 arm_target (변화량 제한용)

        # ── TF2 ──
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # ── 구독 ──
        self.sub_joint = self.create_subscription(
            JointState, '/joint_states', self.joint_state_cb, 10)
        self.sub_cube = self.create_subscription(
            PointStamped, '/cube_position', self.cube_position_cb, 10)

        # ── 시뮬레이터 명령 퍼블리셔 ──
        self.sim_pub = self.create_publisher(
            JointState, SIM_COMMAND_TOPIC, 10)

        # ── 실로봇 액션 클라이언트 (DRY_RUN=False 일 때만 사용) ──
        self.action_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        # ── 제어 루프 ──
        self.timer = self.create_timer(1.0, self.control_loop)
        self.get_logger().info(
            f'시작 — 시뮬 명령 토픽: {SIM_COMMAND_TOPIC}')

    # ─────────────────────────────────────────
    def joint_state_cb(self, msg):
        self.latest_joint_state = msg

    def cube_position_cb(self, msg):
        # z는 학습 분포에 맞춰 고정
        self.latest_cube_pos = np.array(
            [msg.point.x, msg.point.y, CUBE_Z_FIXED], dtype=np.float32)

    # ─────────────────────────────────────────
    def get_ordered_joints(self, msg):
        """ARM_JOINT_NAMES 순서로 재정렬. (그리퍼는 0으로 둠)"""
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        pos = np.zeros(8, dtype=np.float32)
        vel = np.zeros(8, dtype=np.float32)
        for i, jname in enumerate(ARM_JOINT_NAMES):
            if jname not in name_to_idx:
                self.get_logger().warn(
                    f'관절 "{jname}" 없음', throttle_duration_sec=5.0)
                return None, None
            idx = name_to_idx[jname]
            pos[i] = msg.position[idx]
            vel[i] = msg.velocity[idx] if msg.velocity else 0.0
        return pos, vel

    # ─────────────────────────────────────────
    def get_ee_pos(self):
        """world(=base) → link6 변환에 TCP 오프셋 적용. base 기준 좌표."""
        try:
            tf = self.tf_buffer.lookup_transform(
                'world', 'link6', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1))
        except Exception:
            return None
        t = tf.transform.translation
        r = tf.transform.rotation
        pos = np.array([t.x, t.y, t.z], dtype=np.float32)
        quat = np.array([r.w, r.x, r.y, r.z], dtype=np.float32)
        return pos + quat_apply(quat, TCP_LOCAL_OFFSET)

    # ─────────────────────────────────────────
    def build_obs(self):
        msg = self.latest_joint_state
        if msg is None or self.latest_cube_pos is None:
            return None
        pos_8, vel_8 = self.get_ordered_joints(msg)
        if pos_8 is None:
            return None

        joint_pos_rel = pos_8 - DEFAULT_JOINT_POS
        joint_vel_rel = vel_8

        ee_pos = self.get_ee_pos()
        if ee_pos is None:
            self.get_logger().warn('TF world→link6 실패',
                                   throttle_duration_sec=3.0)
            return None

        cube_pos = self.latest_cube_pos
        gripper_pos_rel = pos_8[6:8] - DEFAULT_GRIPPER_POS

        obs = np.concatenate([
            joint_pos_rel, joint_vel_rel, ee_pos, cube_pos, gripper_pos_rel
        ]).reshape(1, -1).astype(np.float32)
        return obs

    # ─────────────────────────────────────────
    def apply_safety(self, arm_target, current_pos):
        """관절 한계 클램핑 + 변화량 제한. 반환: (safe_target, flags)."""
        flags = []

        # 1) 관절 한계 클램핑
        clamped = np.clip(arm_target, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
        if not np.allclose(clamped, arm_target, atol=1e-4):
            flags.append('LIMIT')
        arm_target = clamped

        # 2) 변화량 제한 (직전 명령 기준, 없으면 현재 자세 기준)
        ref = self.prev_target if self.prev_target is not None else current_pos
        delta = arm_target - ref
        if np.any(np.abs(delta) > MAX_DELTA_RAD):
            flags.append('DELTA')
            delta = np.clip(delta, -MAX_DELTA_RAD, MAX_DELTA_RAD)
            arm_target = ref + delta

        return arm_target.astype(np.float32), flags

    # ─────────────────────────────────────────
    def control_loop(self):
        if self.is_executing:
            return

        obs = self.build_obs()
        if obs is None:
            return

        # 정책 추론
        action = self.policy.run(None, {'obs': obs})[0][0]
        arm_action = action[:6]
        arm_target_raw = DEFAULT_ARM_POS + ARM_SCALE * arm_action

        # 안전장치
        current_pos = obs[0, :6] + DEFAULT_ARM_POS  # joint_pos_rel → abs
        arm_target, flags = self.apply_safety(arm_target_raw, current_pos)

        flag_str = f'  [{",".join(flags)}]' if flags else ''
        self.get_logger().info(
            f'target(deg): '
            f'{[f"{math.degrees(v):+.1f}" for v in arm_target]}  '
            f'cube:[{self.latest_cube_pos[0]:.3f},'
            f'{self.latest_cube_pos[1]:.3f},{self.latest_cube_pos[2]:.3f}]'
            f'{flag_str}'
        )

        self.prev_target = arm_target.copy()

        # ── 시뮬레이터로 항상 퍼블리시 ──
        self.publish_to_sim(arm_target)

        # ── 실제 로봇은 DRY_RUN=False 일 때만 ──
        if not DRY_RUN:
            self.send_to_real_robot(arm_target)

    # ─────────────────────────────────────────
    def publish_to_sim(self, arm_target):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = ARM_JOINT_NAMES
        js.position = arm_target.tolist()
        self.sim_pub.publish(js)

    # ─────────────────────────────────────────
    def send_to_real_robot(self, arm_target):
        if not self.action_client.wait_for_server(timeout_sec=0.0):
            self.get_logger().warn('액션 서버 대기 중...',
                                   throttle_duration_sec=5.0)
            return
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ARM_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = arm_target.tolist()
        point.time_from_start = Duration(sec=2, nanosec=0)
        goal.trajectory.points = [point]

        self.is_executing = True
        future = self.action_client.send_goal_async(goal)
        future.add_done_callback(self.goal_response_cb)

    def goal_response_cb(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal 거부됨')
            self.is_executing = False
            return
        goal_handle.get_result_async().add_done_callback(self.result_cb)

    def result_cb(self, future):
        self.is_executing = False


def main():
    rclpy.init()
    node = RLPolicyNodeSim()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
