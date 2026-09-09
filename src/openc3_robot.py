#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
比邻星 Open_C3 协作机械臂 LeRobot 驱动接口 (openc3_robot.py)
===========================================================
【架构说明】
适配比邻星 Open_C3 6-DOF 协作机械臂与电动夹爪：
  1. 继承/遵循 LeRobot 标准 Robot 接口规范 (connect, get_observation, send_action, disconnect)。
  2. 双通信端口机制:
     - 60000 端口: 下发目标关节角与夹爪指令 (Command Socket)。
     - 60001 端口: 接收机械臂当前关节角与力矩状态 (Feedback Socket)。
  3. 支持 Mock 模式与真机 TCP Socket 双模式无缝切换。
"""

import json
import socket
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
import torch

from lerobot.cameras.opencv import OpenCVCameraConfig


@dataclass
class OpenC3RobotConfig:
    """Open_C3 机械臂硬件配置类"""
    robot_ip: str = "192.168.125.1"     # 机械臂出厂默认 IP
    cmd_port: int = 60000               # 指令收发端口
    state_port: int = 60001             # 状态上报端口
    mock: bool = True                   # 是否启用无硬件仿真模式 (SITL)
    cameras: dict[str, OpenCVCameraConfig] = field(default_factory=dict)
    
    # 关节角度物理软限位 (单位: 度)
    joint_limits_deg: list[tuple[float, float]] = field(
        default_factory=lambda: [
            (-175.0, 175.0),  # J1
            (-120.0, 120.0),  # J2
            (-150.0, 150.0),  # J3
            (-175.0, 175.0),  # J4
            (-175.0, 175.0),  # J5
            (-175.0, 175.0),  # J6
        ]
    )
    gripper_limit: tuple[float, float] = (0.0, 100.0)  # 夹爪行程 (0~100)


class OpenC3Robot:
    """
    Open_C3 机械臂驱动类
    """

    def __init__(self, config: OpenC3RobotConfig):
        self.config = config
        self.is_connected = False
        
        # 机械臂 7 维当前状态: [J1, J2, J3, J4, J5, J6 (单位: 弧度), Gripper (0~100)]
        self._current_state = np.zeros(7, dtype=np.float32)
        
        # Socket 客户端句柄
        self._cmd_socket: socket.socket | None = None
        self._state_socket: socket.socket | None = None
        
        # 相机捕获句柄字典
        self._camera_captures: dict[str, cv2.VideoCapture] = {}

    def connect(self):
        """建立机械臂与相机连接"""
        if self.is_connected:
            return

        if self.config.mock:
            print("[OpenC3Robot] 启动 MOCK 虚拟硬件模式 (无需物理机械臂)...")
            # 初始化虚拟位姿 (弧度)
            self._current_state = np.array([0.0, 0.2, -0.5, 0.0, 0.3, 0.0, 50.0], dtype=np.float32)
        else:
            print(f"[OpenC3Robot] 正在连接机械臂 IP: {self.config.robot_ip}...")
            # 1. 连接指令端口 (60000)
            self._cmd_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._cmd_socket.settimeout(2.0)
            self._cmd_socket.connect((self.config.robot_ip, self.config.cmd_port))

            # 2. 连接状态上报端口 (60001)
            self._state_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._state_socket.settimeout(2.0)
            self._state_socket.connect((self.config.robot_ip, self.config.state_port))
            print("[OpenC3Robot] TCP 60000/60001 双端口连接成功！")

        # 初始化挂载的相机设备
        for cam_name, cam_cfg in self.config.cameras.items():
            if self.config.mock:
                print(f"[OpenC3Robot] 挂载虚拟相机: {cam_name} ({cam_cfg.width}x{cam_cfg.height})")
            else:
                cap = cv2.VideoCapture(cam_cfg.index_or_path)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.height)
                cap.set(cv2.CAP_PROP_FPS, cam_cfg.fps)
                self._camera_captures[cam_name] = cap

        self.is_connected = True
        print("[OpenC3Robot] 机械臂与传感器初始化就绪。")

    def get_observation(self) -> dict[str, Any]:
        """
        获取当前传感器原始观测字典
        """
        assert self.is_connected, "机械臂尚未连接，请先调用 connect()"

        # ── 1. 读取关节状态 (统一使用标准键名 'observation.state') ──────
        if self.config.mock:
            state = self._current_state.copy()
        else:
            try:
                data = self._state_socket.recv(1024).decode("utf-8")
                telemetry = json.loads(data)
                joints_rad = np.deg2rad(telemetry["joints_deg"])
                gripper = float(telemetry.get("gripper", 0.0))
                state = np.concatenate([joints_rad, [gripper]]).astype(np.float32)
                self._current_state = state
            except Exception as e:
                print(f"[OpenC3Robot] 状态读取警告: {e}，沿用上一帧状态")
                state = self._current_state.copy()

        # ★ 将原先的 {"state": state} 改为标准键名 {"observation.state": state}
        obs = {"observation.state": state}

        # ── 2. 读取多相机图像 ────────────────────────────────────────
        for cam_name, cam_cfg in self.config.cameras.items():
            if self.config.mock:
                img = np.full((cam_cfg.height, cam_cfg.width, 3), 128, dtype=np.uint8)
            else:
                ret, frame = self._camera_captures[cam_name].read()
                if ret:
                    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    img = np.zeros((cam_cfg.height, cam_cfg.width, 3), dtype=np.uint8)
            obs[cam_name] = img

        return obs

    def send_action(self, action: np.ndarray | dict[str, Any] | torch.Tensor):
        assert self.is_connected, "机械臂尚未连接，请先调用 connect()"

        if isinstance(action, dict):
            # 兼容各种字典 Key 命名
            action = action.get("action", action.get("observation.state", action.get("state", self._current_state)))
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
            
        # 如果经过 Tensor 解包后是 (1, 7)，展平成 (7,)
        if action.ndim > 1:
            action = action.squeeze()

        target_joints_rad = action[:6]
        target_gripper = float(action[6])
        
        target_joints_deg = np.rad2deg(target_joints_rad)
        for i, (min_deg, max_deg) in enumerate(self.config.joint_limits_deg):
            target_joints_deg[i] = np.clip(target_joints_deg[i], min_deg, max_deg)
        target_gripper = np.clip(target_gripper, *self.config.gripper_limit)

        if self.config.mock:
            alpha = 0.2
            current_rad = self._current_state[:6]
            self._current_state[:6] = current_rad + alpha * (np.deg2rad(target_joints_deg) - current_rad)
            self._current_state[6] = self._current_state[6] + alpha * (target_gripper - self._current_state[6])
        else:
            cmd_payload = {
                "command": "move_joint",
                "joints_deg": target_joints_deg.tolist(),
                "gripper": target_gripper,
                "timestamp": time.time(),
            }
            try:
                msg = json.dumps(cmd_payload).encode("utf-8")
                self._cmd_socket.sendall(msg)
            except Exception as e:
                print(f"[OpenC3Robot] 动作指令下发失败: {e}")

    def disconnect(self):
        """断开连接并释放端口和相机"""
        if not self.is_connected:
            return

        if not self.config.mock:
            if self._cmd_socket:
                self._cmd_socket.close()
            if self._state_socket:
                self._state_socket.close()

        for cap in self._camera_captures.values():
            cap.release()

        self._camera_captures.clear()
        self.is_connected = False
        print("[OpenC3Robot] 连接已断开，资源释放完毕。")