import os

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
from typing import Dict, Tuple
import mujoco
import numpy as np
import torch

DEFAULT_XML_PATH = str(Path(__file__).resolve().parent.parent / "assets" / "openc3_scene.xml")


class OpenC3PushTEnv:
    """Open_C3 机械臂 3D Push-T 物理仿真环境。"""

    def __init__(
        self,
        xml_path: str = DEFAULT_XML_PATH,
        control_freq: float = 30.0,
        image_size: Tuple[int, int] = (224, 224),
    ):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"MJCF 文件未找到: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)

        self.control_freq = control_freq
        self.sim_timestep = self.model.opt.timestep  # 0.002s
        self.n_substeps = int(1.0 / (control_freq * self.sim_timestep))

        self.img_h, self.img_w = image_size
        self.renderer = mujoco.Renderer(self.model, height=self.img_h, width=self.img_w)

        # 关节与执行器 ID 映射
        self.joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_j{i}") for i in range(1, 7)]
        self.tool_act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_tool")
        self.tee_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "tee_joint")
        self.pusher_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "pusher_site")

        # 提取机械臂专属的 qpos 与 dof 索引 (避开 freejoint)
        self.joint_qpos_indices = [self.model.jnt_qposadr[j_id] for j_id in self.joint_ids]
        self.joint_dof_indices = [self.model.jnt_dofadr[j_id] for j_id in self.joint_ids]

        self.goal_pos_2d = np.array([0.38, 0.0], dtype=np.float32)
        self.reset()

    def reset(self, random_tee: bool = False) -> Dict[str, torch.Tensor]:
        mujoco.mj_resetData(self.model, self.data)

        # 1. 设置机械臂初始姿态 (更新为 CAD 标定值)
        self.home_qpos = np.array(
            [-0.12, 0.46, 2.08, -1.05, -1.57, 0.0], dtype=np.float32
        )
        for i, adr in enumerate(self.joint_qpos_indices):
            self.data.qpos[adr] = self.home_qpos[i]
            
        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = self.home_qpos[i]

        # 2. 设置方块初始位置 (增加最小距离约束与适度初始偏角)
        tee_qpos_addr = self.model.jnt_qposadr[self.tee_joint_id]
        if random_tee:
            while True:
                rand_x = float(np.random.uniform(0.32, 0.40))
                rand_y = float(np.random.uniform(-0.065, 0.065))
                # 确保物块初始距目标至少 35mm，防止开局直接误判达标
                if np.linalg.norm(np.array([rand_x, rand_y]) - self.goal_pos_2d) >= 0.035:
                    break
            self.data.qpos[tee_qpos_addr : tee_qpos_addr + 3] = [rand_x, rand_y, 0.4155]
            
            # 随机注入初始偏航角 [-15°, +15°]
            rand_yaw = float(np.random.uniform(-0.26, 0.26))
            self.data.qpos[tee_qpos_addr + 3 : tee_qpos_addr + 7] = [
                np.cos(rand_yaw / 2.0), 0.0, 0.0, np.sin(rand_yaw / 2.0)
            ]
        else:
            self.data.qpos[tee_qpos_addr : tee_qpos_addr + 3] = [0.35, -0.05, 0.4155]
            self.data.qpos[tee_qpos_addr + 3 : tee_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]

        # ==========================================
        # 【核心新增：绝对静止锁】
        # 强制将整个物理系统所有关节与自由度的“速度 (qvel)”全部归零！
        # 彻底根除上一轮结尾残余动量带入下一轮导致的开局爆炸。
        # ==========================================
        self.data.qvel[:] = 0.0

        # 向前推进一步，让 MuJoCo 刷新动力学矩阵
        mujoco.mj_forward(self.model, self.data)

        return self._get_observation()

    def get_pusher_pos(self) -> np.ndarray:
        """获取推杆尖端当前笛卡尔三维坐标。"""
        return self.data.site_xpos[self.pusher_site_id].copy()

    def get_tee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """获取 T-Block 的三维坐标与四元数。"""
        tee_qpos_addr = self.model.jnt_qposadr[self.tee_joint_id]
        pos = self.data.qpos[tee_qpos_addr : tee_qpos_addr + 3].copy()
        quat = self.data.qpos[tee_qpos_addr + 3 : tee_qpos_addr + 7].copy()
        return pos, quat

    def ik_solve_position(
        self, 
        target_pos: np.ndarray, 
        max_iters: int = 60, 
        tol_pos: float = 1e-3, 
        tol_tilt_rad: float = 0.01
    ) -> np.ndarray:
        """5-DOF 轴对称垂直阻尼最小二乘法 IK：保证推杆绝对垂直 (倾角 0.000°) 且毫米级追踪目标"""
        saved_qpos = self.data.qpos.copy()
        q = np.array([self.data.qpos[adr] for adr in self.joint_qpos_indices], dtype=np.float64)
        jnt_ranges = np.array([self.model.jnt_range[j_id] for j_id in self.joint_ids])
        
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        target_dir = np.array([0.0, 0.0, -1.0], dtype=np.float64)

        damping = 1e-4
        w_tilt = 0.4  # 倾角约束权重
        k_null = 0.3  # 零空间拉回 Home 姿态系数

        for _ in range(max_iters):
            for i, adr in enumerate(self.joint_qpos_indices):
                self.data.qpos[adr] = q[i]
            mujoco.mj_forward(self.model, self.data)

            # 1. 位置误差 (3D)
            curr_pos = self.data.site_xpos[self.pusher_site_id]
            err_pos = target_pos - curr_pos

            # 2. 倾斜误差 (仅约束 X/Y 偏转，释放轴对称自转)
            pusher_dir = -self.data.site_xmat[self.pusher_site_id].reshape(3, 3)[:, 0]
            err_cross = np.cross(pusher_dir, target_dir)
            err_tilt = err_cross[:2]

            if np.linalg.norm(err_pos) < tol_pos and np.linalg.norm(err_tilt) < tol_tilt_rad:
                break

            # 3. 构造 5x6 雅可比矩阵
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.pusher_site_id)
            jac_p = jacp[:, self.joint_dof_indices]
            jac_tilt = jacr[:2, env_dof := self.joint_dof_indices]

            J = np.vstack([jac_p, w_tilt * jac_tilt])
            err = np.concatenate([err_pos, w_tilt * err_tilt])

            # 4. 阻尼伪逆求解与零空间投影
            J_JT = J @ J.T + (damping ** 2) * np.eye(5)
            J_dagger = J.T @ np.linalg.inv(J_JT)

            dq_main = J_dagger @ err
            null_proj = np.eye(6) - J_dagger @ J
            dq_null = null_proj @ (self.home_qpos - q)

            dq = dq_main + k_null * dq_null
            q += np.clip(dq, -0.12, 0.12)
            q = np.clip(q, jnt_ranges[:, 0], jnt_ranges[:, 1])

        self.data.qpos[:] = saved_qpos
        mujoco.mj_forward(self.model, self.data)
        return q.astype(np.float32)

    def step(self, action: np.ndarray) -> Dict[str, torch.Tensor]:
        assert len(action) == 7, f"Action 维度应为 7，当前为 {len(action)}"

        target_arm_qpos = action[:6].copy()
        
        # 获取当前的真实控制指令
        current_ctrl = np.array([self.data.ctrl[act_id] for act_id in self.actuator_ids])
        
        target_tool_ctrl = (action[6] / 100.0) * 0.01
        current_tool_ctrl = self.data.ctrl[self.tool_act_id]

        # 彻底移除 max_delta 暴力限速！
        # 完全信任 FSM 规划的密集平滑轨迹，仅使用子步插值消除离散控制的扭矩阶跃。
        for step_i in range(1, self.n_substeps + 1):
            alpha = step_i / self.n_substeps
            
            # 子步级平滑过渡：从当前指令平滑逼近目标指令
            interp_ctrl = (1.0 - alpha) * current_ctrl + alpha * target_arm_qpos
            for i, act_id in enumerate(self.actuator_ids):
                self.data.ctrl[act_id] = interp_ctrl[i]
                
            self.data.ctrl[self.tool_act_id] = (1.0 - alpha) * current_tool_ctrl + alpha * target_tool_ctrl
            
            mujoco.mj_step(self.model, self.data)

        return self._get_observation()

    def _render_camera(self, camera_name: str) -> torch.Tensor:
        self.renderer.update_scene(self.data, camera=camera_name)
        rgb = self.renderer.render()
        return torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float() / 255.0

    def _get_observation(self) -> Dict[str, torch.Tensor]:
        joint_pos = np.array(
            [self.data.qpos[adr] for adr in self.joint_qpos_indices],
            dtype=np.float32,
        )
        state_7d = np.concatenate([joint_pos, [0.0]], dtype=np.float32)

        return {
            "observation.state": torch.from_numpy(state_7d),
            "observation.images.top": self._render_camera("top_cam"),
            "observation.images.wrist": self._render_camera("wrist_cam"),
        }