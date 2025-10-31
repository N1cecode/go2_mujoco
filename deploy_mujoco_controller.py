#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import sys
import math
import argparse
import yaml
import numpy as np
import torch
import mujoco
import mujoco.viewer
import ctypes
from ctypes import Structure, c_ushort, c_short, c_ulong, byref

# ----------------- XInput（ctypes, 零依赖） -----------------
class _XInputWrap:
    def __init__(self):
        self.lib = None
        for dll in ("xinput1_4.dll", "xinput1_3.dll", "xinput9_1_0.dll"):
            try:
                self.lib = ctypes.windll.LoadLibrary(dll)
                break
            except OSError:
                self.lib = None
        if self.lib is None:
            raise OSError("未找到 XInput DLL（xinput1_4/1_3/9_1_0）")

        class XINPUT_GAMEPAD(Structure):
            _fields_ = [
                ("wButtons", c_ushort),
                ("bLeftTrigger", ctypes.c_ubyte),
                ("bRightTrigger", ctypes.c_ubyte),
                ("sThumbLX", c_short),
                ("sThumbLY", c_short),
                ("sThumbRX", c_short),
                ("sThumbRY", c_short),
            ]
        class XINPUT_STATE(Structure):
            _fields_ = [("dwPacketNumber", c_ulong), ("Gamepad", XINPUT_GAMEPAD)]
        class XINPUT_VIBRATION(Structure):
            _fields_ = [("wLeftMotorSpeed", c_ushort), ("wRightMotorSpeed", c_ushort)]

        self.XINPUT_STATE = XINPUT_STATE
        self.XINPUT_VIBRATION = XINPUT_VIBRATION

        self.XInputGetState = self.lib.XInputGetState
        self.XInputGetState.argtypes = (c_ulong, ctypes.POINTER(XINPUT_STATE))
        self.XInputGetState.restype = ctypes.c_ulong

        self.XInputSetState = self.lib.XInputSetState
        self.XInputSetState.argtypes = (c_ulong, ctypes.POINTER(XINPUT_VIBRATION))
        self.XInputSetState.restype = ctypes.c_ulong

        # button masks
        self.DPAD_UP        = 0x0001
        self.DPAD_DOWN      = 0x0002
        self.DPAD_LEFT      = 0x0004
        self.DPAD_RIGHT     = 0x0008
        self.START          = 0x0010
        self.BACK           = 0x0020
        self.L_THUMB        = 0x0040
        self.R_THUMB        = 0x0080
        self.LB             = 0x0100
        self.RB             = 0x0200
        self.A              = 0x1000
        self.B              = 0x2000
        self.X              = 0x4000
        self.Y              = 0x8000

    def poll(self, idx=0):
        """返回 dict 或 None（未连接）"""
        st = self.XINPUT_STATE()
        if self.XInputGetState(idx, byref(st)) != 0:
            return None
        g = st.Gamepad

        def n16(v):
            v = float(np.clip(v, -32767, 32767))
            return v / 32767.0

        btns = g.wButtons
        return {
            "buttons": btns,
            "A": bool(btns & self.A),
            "B": bool(btns & self.B),
            "X": bool(btns & self.X),
            "Y": bool(btns & self.Y),
            "LB": bool(btns & self.LB),
            "RB": bool(btns & self.RB),
            "DPAD_UP":    bool(btns & self.DPAD_UP),
            "DPAD_DOWN":  bool(btns & self.DPAD_DOWN),
            "DPAD_LEFT":  bool(btns & self.DPAD_LEFT),
            "DPAD_RIGHT": bool(btns & self.DPAD_RIGHT),
            "LT": g.bLeftTrigger / 255.0,
            "RT": g.bRightTrigger / 255.0,
            "LX": n16(g.sThumbLX),
            "LY": n16(g.sThumbLY),
            "RX": n16(g.sThumbRX),
            "RY": n16(g.sThumbRY),
        }

    def vibrate(self, idx=0, left=0.0, right=0.0):
        vib = self.XINPUT_VIBRATION()
        vib.wLeftMotorSpeed  = int(np.clip(left, 0, 1) * 65535)
        vib.wRightMotorSpeed = int(np.clip(right, 0, 1) * 65535)
        self.XInputSetState(idx, byref(vib))


# ----------------- 工具函数 -----------------
def get_gravity_orientation(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """四元数 [w,x,y,z] -> 机体坐标系下的重力方向 (3,)"""
    qw, qx, qy, qz = float(quaternion_wxyz[0]), float(quaternion_wxyz[1]), float(quaternion_wxyz[2]), float(quaternion_wxyz[3])
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2.0 * (-qz * qx + qw * qy)
    g[1] = -2.0 * (qz * qy + qw * qx)
    g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
    return g


def quat_wxyz_to_yaw(q):
    """从四元数 [w,x,y,z] 提取世界坐标航向角 yaw（弧度）"""
    w, x, y, z = q
    s = 2.0 * (w * z + x * y)
    c = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(s, c)


def pd_control(target_q: np.ndarray, q: np.ndarray, kp: np.ndarray,
               target_dq: np.ndarray, dq: np.ndarray, kd: np.ndarray) -> np.ndarray:
    return (target_q - q) * kp + (target_dq - dq) * kd


def _apply_deadzone_curve(x: float, dz: float, expo: float) -> float:
    """死区 + 指数曲线（expo∈[0,1..]，0=线性）"""
    if abs(x) < dz: return 0.0
    x = np.sign(x) * (abs(x) - dz) / (1.0 - dz)
    return float(np.sign(x) * (abs(x) ** (1.0 + expo)))


def _print_status_once(lx, ly, rx, speed_scale, cmd_vec):
    """单行覆盖式输出当前摇杆与指令（不刷屏），并返回同样的字符串以供 viewer 显示。"""
    vx, vy, yaw = cmd_vec.tolist()
    msg = (f"[JOY] LX={lx:+.2f} LY={ly:+.2f} RX={rx:+.2f}  "
           f"s={speed_scale:.2f}  "
           f"cmd: vx={vx:+.2f} m/s  vy={vy:+.2f} m/s  yaw={yaw:+.2f} rad/s")
    sys.stdout.write("\r" + msg + " " * 8)
    sys.stdout.flush()
    return msg


# ----------------- 主程序 -----------------
def main():
    # 解析配置
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", type=str, default="go2.yaml", help="配置文件名（位于 ./configs/ 下）")
    args = parser.parse_args()

    cfg_path = f"./configs/{args.config_file}"
    with open(cfg_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    policy_path = config["policy_path"]
    xml_path = config["xml_path"]

    simulation_duration: float = float(config["simulation_duration"])
    simulation_dt: float = float(config["simulation_dt"])
    control_decimation: int = int(config["control_decimation"])

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    ang_vel_scale: float = float(config["ang_vel_scale"])
    dof_pos_scale: float = float(config["dof_pos_scale"])
    dof_vel_scale: float = float(config["dof_vel_scale"])
    action_scale: float = float(config["action_scale"])

    # 摇杆→指令：乘法语义；观测直接用 cmd（不再乘一次）
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)    # (3,)
    cmd = np.array(config["cmd_init"], dtype=np.float32)           # (3,)

    num_actions: int = int(config["num_actions"])
    num_obs: int = int(config["num_obs"])
    assert num_actions == 12 and num_obs == 45
    assert kps.shape[0] == 12 and kds.shape[0] == 12 and default_angles.shape[0] == 12

    # 手柄配置（带默认）
    jcfg = dict(
        enabled=bool(config.get("joystick", {}).get("enabled", True)),
        deadzone=float(config.get("joystick", {}).get("deadzone", 0.12)),
        expo=float(config.get("joystick", {}).get("expo", 0.35)),
        speed_scale=float(config.get("joystick", {}).get("speed_scale_init", 1.0)),
        speed_step=float(config.get("joystick", {}).get("speed_step", 0.25)),
    )
    # 相机参数
    cam_cfg = dict(
        follow=config.get("camera", {}).get("follow", True),
        distance=float(config.get("camera", {}).get("distance", 2.5)),      # 跟随距离
        elevation=float(config.get("camera", {}).get("elevation_deg", -20.0)),  # 俯视角（度）：负数=从上往下看
        azimuth_offset=float(config.get("camera", {}).get("azimuth_offset_deg", 0.0)),
    )

    # MuJoCo 模型与初始态
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    # 记录初始状态用于 reset
    qpos0 = d.qpos.copy()
    qvel0 = d.qvel.copy()

    # 关节->电机映射（用于写入 d.ctrl）
    joint2motor_idx = np.array(
        config.get("joint2motor_idx", list(range(num_actions))),
        dtype=np.int32
    )
    assert joint2motor_idx.shape[0] == num_actions, \
        f"joint2motor_idx 长度应为 {num_actions}，实际 {joint2motor_idx.shape[0]}"
    assert m.nu >= num_actions, \
        f"模型电机数 m.nu={m.nu} 小于关节数 {num_actions}"
    assert np.all((0 <= joint2motor_idx) & (joint2motor_idx < m.nu)), \
        "joint2motor_idx 含越界索引"

    # 找到“基座 body”用于相机跟随（优先采用 free joint 所在的 body）
    if m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE and m.jnt_qposadr[0] == 0:
        base_body_id = int(m.jnt_bodyid[0])
    else:
        base_body_id = 0  # 兜底

    # 策略
    policy = torch.jit.load(policy_path, map_location="cpu")
    policy.eval()

    # 上下文
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    counter = 0
    action_smoothing = float(config.get("action_smoothing", 0.0))
    action_smoothed = action.copy()

    # 手柄
    xpad = None
    prev_buttons = 0  # 用于边沿触发（重置、D-pad）
    if jcfg["enabled"]:
        try:
            xpad = _XInputWrap()
            if xpad.poll(0) is None:
                print("[Joystick] 未检测到 XInput 控制器（idx=0）。将继续运行（保持 cmd_init）。")
            else:
                print("[Joystick] 已连接 XInput 控制器（idx=0）。左摇杆: vx/vy，右摇杆: yaw，D-pad 调速，A 刹停，Y 重置。")
        except Exception as e:
            print(f"[Joystick] 初始化失败：{e}。将继续运行（保持 cmd_init）。")
            xpad = None

    def do_reset():
        nonlocal d, qpos0, qvel0, target_dof_pos, action, action_smoothed, cmd, counter
        d.qpos[:] = qpos0
        d.qvel[:] = qvel0
        mujoco.mj_forward(m, d)
        target_dof_pos = default_angles.copy()
        action[:] = 0.0
        action_smoothed[:] = 0.0
        cmd[:] = 0.0
        counter = 0
        print("\n[RESET] 状态已重置。")

    def update_cmd_from_joystick(cmd_vec: np.ndarray):
        """
        从手柄刷新 cmd=[vx, vy, yaw]（乘法语义）：
        cmd = speed_scale * [ ly, -lx, -rx ] * cmd_scale
             ↑ vx 正向：LY 上推为 +；  vy/yaw 方向按你的反馈取反（-lx、-rx）
        返回：(cmd, st, lx, ly, rx, buttons)
        """
        nonlocal jcfg, xpad, cmd_scale
        st = None if xpad is None else xpad.poll(0)
        if st is None:
            return cmd_vec, None, 0.0, 0.0, 0.0, 0

        # 处理后轴值（应用死区/曲线）
        lx = _apply_deadzone_curve(st["LX"], jcfg["deadzone"], jcfg["expo"])
        ly = _apply_deadzone_curve(st["LY"], jcfg["deadzone"], jcfg["expo"])
        rx = _apply_deadzone_curve(st["RX"], jcfg["deadzone"], jcfg["expo"])

        # 乘法增益；并修正 vy / yaw 方向（取负）
        vx  =  ly * jcfg["speed_scale"] * cmd_scale[0]   # 上推为 +vx
        vy  = -lx * jcfg["speed_scale"] * cmd_scale[1]   # 反向修正
        yaw = -rx * jcfg["speed_scale"] * cmd_scale[2]   # 反向修正

        return np.array([vx, vy, yaw], dtype=np.float32), st, lx, ly, rx, int(st["buttons"])

    # MuJoCo viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        # 初始相机放到后上方
        viewer.cam.distance = cam_cfg["distance"]
        viewer.cam.elevation = cam_cfg["elevation"]

        start = time.time()
        try:
            while viewer.is_running() and (time.time() - start < simulation_duration):
                step_start = time.time()

                # 低级 PD
                tau = pd_control(
                    target_q=target_dof_pos, q=d.qpos[7:], kp=kps,
                    target_dq=np.zeros_like(kds), dq=d.qvel[6:], kd=kds
                )

                # === 关节->电机 映射写入 ===
                d.ctrl[:] = 0.0
                d.ctrl[joint2motor_idx] = tau

                mujoco.mj_step(m, d)
                counter += 1

                # 控制周期：策略 & 摇杆
                if counter % control_decimation == 0:
                    # 读取状态
                    qj = d.qpos[7:]
                    dqj = d.qvel[6:]
                    quat = d.qpos[3:7]
                    omega = d.qvel[3:6]

                    # 归一化/缩放
                    qj_n = (qj - default_angles) * dof_pos_scale
                    dqj_n = dqj * dof_vel_scale
                    gravity_orientation = get_gravity_orientation(quat)
                    omega_n = omega * ang_vel_scale

                    # 手柄更新
                    cmd, st, lx_show, ly_show, rx_show, buttons = update_cmd_from_joystick(cmd)

                    # --- 边沿触发：D-pad 调速 & Y 重置 ---
                    if xpad is not None:
                        def rising(mask): return (buttons & mask) and not (prev_buttons & mask)
                        if rising(xpad.DPAD_UP):
                            jcfg["speed_scale"] = float(np.clip(jcfg["speed_scale"] + jcfg["speed_step"], 0.1, 2.0))
                        if rising(xpad.DPAD_DOWN):
                            jcfg["speed_scale"] = float(np.clip(jcfg["speed_scale"] - jcfg["speed_step"], 0.05, 2.0))
                        if rising(xpad.DPAD_LEFT):
                            jcfg["speed_scale"] = 0.5
                        if rising(xpad.DPAD_RIGHT):
                            jcfg["speed_scale"] = 1.0
                        if rising(xpad.Y):      # Y 键重置
                            do_reset()
                            prev_buttons = buttons
                            viewer.sync()
                            continue
                        prev_buttons = buttons

                    # 组装 45 维观测（直接送入 cmd）
                    obs[:3] = omega_n
                    obs[3:6] = gravity_orientation
                    obs[6:9] = cmd
                    obs[9:21] = qj_n
                    obs[21:33] = dqj_n
                    obs[33:45] = action_smoothed

                    # 策略推理
                    obs_tensor = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0)
                    with torch.no_grad():
                        act = policy(obs_tensor).cpu().numpy().squeeze().astype(np.float32)

                    # 平滑
                    if action_smoothing > 0.0:
                        action_smoothed = action_smoothing * action_smoothed + (1.0 - action_smoothing) * act
                    else:
                        action_smoothed = act
                    action = act

                    # 动作 -> 目标关节角
                    target_dof_pos = action_smoothed * action_scale + default_angles

                    # 状态行（终端 & viewer）
                    status_line = _print_status_once(lx_show, ly_show, rx_show, jcfg["speed_scale"], cmd)
                    viewer.user_warning = status_line

                # 相机跟随（每步）
                if cam_cfg["follow"]:
                    base_pos = d.xpos[base_body_id].copy()
                    viewer.cam.lookat[:] = base_pos
                    yaw_rad = quat_wxyz_to_yaw(d.qpos[3:7])
                    # —— 关键：后上方 —— 相机方位 = 航向
                    viewer.cam.azimuth = math.degrees(yaw_rad) + cam_cfg["azimuth_offset"]
                    viewer.cam.elevation = cam_cfg["elevation"]
                    viewer.cam.distance = cam_cfg["distance"]

                viewer.sync()

                # 时间对齐
                time_until_next_step = m.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        finally:
            print()  # 结束时换行，避免光标停在单行输出

if __name__ == "__main__":
    main()
