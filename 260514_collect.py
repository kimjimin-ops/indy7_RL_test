"""
Isaac Sim Standalone Teleop — 직접 조인트 제어 방식 및 보상 기반 데이터 수집 (스냅샷 방식)
실행: /home/kimjimin/IsaacLab/isaaclab.sh -p /home/kimjimin/Desktop/teleop_collect.py
"""

import argparse
import math
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args, _ = parser.parse_known_args()
args.headless = False
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch
import numpy as np
import h5py
import os
from pynput import keyboard as kb

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg

# Quaternion 연산용 유틸리티 임포트
from isaaclab.utils.math import quat_apply

##############################################################
# 상수
##############################################################
USD_PATH     = "/home/kimjimin/test3/source/test3/test3/assets/model/usd/indy7/indy7_simplified_260424.usda"
SAVE_PATH    = "/home/kimjimin/Desktop/joint0_demos.hdf5"
GRIPPER_OPEN   = 0.015
GRIPPER_CLOSED = 0.0
JOINT_STEP     = 0.02   # 조인트 이동 단위 (rad)

##############################################################
# 키보드 상태
##############################################################
selected_joint = 0        # 현재 선택된 조인트 (0~5)
joint_direction = 0       # +1, -1, 0
gripper_state  = GRIPPER_OPEN
save_episode   = False
quit_flag      = False

pressed_keys = set()

def on_press(key):
    global selected_joint, joint_direction
    global gripper_state, save_episode, quit_flag

    try:
        ch = key.char
        if ch is None:
            return

        pressed_keys.add(ch)

        if ch in '123456':
            selected_joint = int(ch) - 1
            print(f"[선택] Joint {selected_joint} (joint{selected_joint})")

        if ch == '+' or ch == '=':
            joint_direction = 1
        if ch == '-':
            joint_direction = -1

        if ch == 'g':
            gripper_state = GRIPPER_CLOSED if gripper_state == GRIPPER_OPEN \
                            else GRIPPER_OPEN
            label = "CLOSED (grasp)" if gripper_state == GRIPPER_CLOSED else "OPEN"
            print(f"[Gripper] {label}")

        if ch == 'r':
            save_episode = True

    except AttributeError:
        if key == kb.Key.esc:
            quit_flag = True

def on_release(key):
    global joint_direction
    try:
        ch = key.char
        pressed_keys.discard(ch)
        if ch in ('+', '=', '-'):
            joint_direction = 0
    except AttributeError:
        pass

listener = kb.Listener(on_press=on_press, on_release=on_release)
listener.start()

##############################################################
# SimulationContext
##############################################################
sim_cfg = sim_utils.SimulationCfg(
    dt=1.0 / 60.0,
    device="cpu",
)
sim = SimulationContext(sim_cfg)
sim.set_camera_view(eye=[1.5, 1.5, 1.5], target=[0.5, 0.0, 0.5])

##############################################################
# 씬 구성
##############################################################
sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2500.0).func(
    "/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

cube_cfg = RigidObjectCfg(
    prim_path="/World/Cube",
    spawn=sim_utils.CuboidCfg(
        size=(0.07, 0.07, 0.07),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.1),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.025)),
)
cube = RigidObject(cfg=cube_cfg)

robot_cfg = ArticulationCfg(
    prim_path="/World/indy7",
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint0": 0.0,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "PrismaticJoint":           GRIPPER_OPEN,
            "PrismaticJoint_finger1_b": GRIPPER_OPEN,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[0-5]"],
            stiffness=400.0,
            damping=40.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["PrismaticJoint", "PrismaticJoint_finger1_b"],
            stiffness=10000.0,
            damping=100.0,
        ),
    },
)
robot = Articulation(cfg=robot_cfg)

##############################################################
# 초기화
##############################################################
sim.reset()
robot.update(sim.cfg.dt)
cube.update(sim.cfg.dt)

joint_names   = robot.data.joint_names
gripper_ids   = [joint_names.index("PrismaticJoint"),
                 joint_names.index("PrismaticJoint_finger1_b")]
arm_ids       = [joint_names.index(f"joint{i}") for i in range(6)]

# env_cfg 보상에서 사용되는 링크 인덱스 탐색
try:
    link6_idx = robot.data.body_names.index("link6")
except ValueError:
    link6_idx = 0
    print("[경고] link6를 찾을 수 없습니다.")

try:
    finger_idx = robot.data.body_names.index("fingerhalf_01")
except ValueError:
    finger_idx = link6_idx  # 없으면 link6로 대체

joint_target  = robot.data.joint_pos.clone()   # [1, 8]
JOINT_LIMIT = np.deg2rad(175.0)

##############################################################
# 데이터 버퍼
##############################################################
all_snapshots = []
episode_count = 0
TARGET_DEMOS  = 20

print("\n=== 조작 방법 ===")
print("  1~6    : 조인트 선택 (joint0~5)")
print("  +      : 선택 조인트 + 방향")
print("  -      : 선택 조인트 - 방향")
print("  G      : 그리퍼 토글 (열림 ↔ 닫힘)")
print("  R      : 현재 상태 스냅샷 저장 (보상 데이터 포함)")
print("  ESC    : 종료")
print(f"\n현재 선택: Joint 0\n")

##############################################################
# 메인 루프
##############################################################
while simulation_app.is_running() and not quit_flag:

    # ── 1. 선택 조인트 이동 ──────────────────────────────
    if joint_direction != 0:
        idx = arm_ids[selected_joint]
        joint_target[0, idx] += joint_direction * JOINT_STEP
        # 한계 클리핑
        joint_target[0, idx] = torch.clamp(
            joint_target[0, idx],
            min=-JOINT_LIMIT, max=JOINT_LIMIT
        )

    # ── 2. 그리퍼 ─────────────────────────────────────
    joint_target[0, gripper_ids[0]] = gripper_state
    joint_target[0, gripper_ids[1]] = gripper_state

    # ── 3. 명령 적용 ──────────────────────────────────
    robot.set_joint_position_target(joint_target)
    robot.write_data_to_sim()

    # ── 4. 시뮬레이션 스텝 ───────────────────────────
    sim.step()
    robot.update(sim.cfg.dt)
    cube.update(sim.cfg.dt)

    # ── 5. R키: 현재 스냅샷 저장 ───────────────────────
    if save_episode:
        save_episode = False
        device = sim.device
        
        # 현재 상태 추출
        obs_vec    = robot.data.joint_pos[0].cpu().numpy().copy()
        action_vec = joint_target[0].cpu().numpy().copy()
        
        # Link 정보 및 큐브 정보 추출
        link6_pos  = robot.data.body_state_w[0, link6_idx, :3]
        link6_quat = robot.data.body_state_w[0, link6_idx, 3:7]
        finger_pos = robot.data.body_state_w[0, finger_idx, :3]
        
        cube_pos  = cube.data.root_state_w[0, :3]
        cube_quat = cube.data.root_state_w[0, 3:7]
        
        # [A] cube_lifted 검사
        lift_threshold = 0.1
        cube_z = cube_pos[2].item()
        is_lifted = 1.0 if cube_z > lift_threshold else 0.0
        
        # [B] gripper_auto_reward 계산 (finger_idx 사용, close_dist=0.15)
        dist_finger_cube = torch.norm(finger_pos - cube_pos).item()
        target_grip = 0.0 if dist_finger_cube <= 0.15 else 0.015
        
        grip_pos_avg = robot.data.joint_pos[0, gripper_ids].mean().item()
        grip_error = abs(grip_pos_avg - target_grip)
        gripper_reward = 1.0 - math.tanh(grip_error / 0.005)

        # [C] ee_to_cube_distance 계산 (link6 tcp 기반 offset 반영)
        link6_pos_b  = link6_pos.unsqueeze(0)
        link6_quat_b = link6_quat.unsqueeze(0)
        cube_pos_b   = cube_pos.unsqueeze(0)
        
        local_offset = torch.tensor([[0.0, 0.0, 0.27]], device=device)
        world_offset = quat_apply(link6_quat_b, local_offset)
        tcp_pos = link6_pos_b + world_offset
        
        tcp_dist = torch.norm(tcp_pos - cube_pos_b, dim=1).item()
        tcp_reward = 1.0 - math.tanh(tcp_dist / 0.5)

        # [D] ee_cube_axis_alignment 계산 (link6 vs cube)
        cube_quat_b = cube_quat.unsqueeze(0)
        
        x_unit = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        y_unit = torch.tensor([[0.0, 1.0, 0.0]], device=device)

        ee_x = quat_apply(link6_quat_b, x_unit)
        ee_y = quat_apply(link6_quat_b, y_unit)
        c_x  = quat_apply(cube_quat_b, x_unit)
        c_y  = quat_apply(cube_quat_b, y_unit)

        align_x = torch.sum(ee_x * c_x, dim=1)
        align_y = torch.sum(ee_y * c_y, dim=1)
        axis_align_reward = ((torch.abs(align_x) + torch.abs(align_y)) / 2.0).item()

        # 리스트로 래핑하여 차원을 유지 (1스텝짜리 데이터셋)
        snapshot_data = {
            "obs": [obs_vec],
            "actions": [action_vec],
            "cube_lifted": [is_lifted],
            "finger_cube_dist": [dist_finger_cube],
            "gripper_target": [target_grip],
            "gripper_reward": [gripper_reward],
            "tcp_cube_dist": [tcp_dist],
            "tcp_reward": [tcp_reward],
            "axis_align_reward": [axis_align_reward]
        }

        all_snapshots.append(snapshot_data)
        episode_count += 1
        
        print(f"[저장 {episode_count}/{TARGET_DEMOS}] Lift:{is_lifted} | Grip_Rew:{gripper_reward:.2f} | TCP_Rew:{tcp_reward:.2f} | Align_Rew:{axis_align_reward:.2f}")

    if episode_count >= TARGET_DEMOS:
        print("\n[완료] 목표 스냅샷 수 달성")
        break
    

##############################################################
# HDF5 저장
##############################################################
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

with h5py.File(SAVE_PATH, "w") as f:
    grp = f.create_group("data")
    for i, snap in enumerate(all_snapshots):
        ep_grp = grp.create_group(f"demo_{i}")
        ep_grp.create_dataset("obs", data=np.array(snap["obs"], dtype=np.float32))
        ep_grp.create_dataset("actions", data=np.array(snap["actions"], dtype=np.float32))
        
        # 스냅샷 환경 보상/수치 지표들
        ep_grp.create_dataset("cube_lifted", data=np.array(snap["cube_lifted"], dtype=np.float32))
        ep_grp.create_dataset("finger_cube_dist", data=np.array(snap["finger_cube_dist"], dtype=np.float32))
        ep_grp.create_dataset("gripper_target", data=np.array(snap["gripper_target"], dtype=np.float32))
        ep_grp.create_dataset("gripper_reward", data=np.array(snap["gripper_reward"], dtype=np.float32))
        ep_grp.create_dataset("tcp_cube_dist", data=np.array(snap["tcp_cube_dist"], dtype=np.float32))
        ep_grp.create_dataset("tcp_reward", data=np.array(snap["tcp_reward"], dtype=np.float32))
        ep_grp.create_dataset("axis_align_reward", data=np.array(snap["axis_align_reward"], dtype=np.float32))
        
        ep_grp.attrs["num_samples"] = 1 # 스냅샷 1개이므로 고정값 1
    
    grp.attrs["total_demos"] = len(all_snapshots)
    print(f"\n[성공] HDF5 파일 저장 완료: {SAVE_PATH}")
