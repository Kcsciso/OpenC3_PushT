import os
os.environ["DISPLAY"] = ":1"
os.environ["MUJOCO_GL"] = "glfw"

import time
import numpy as np
import mujoco.viewer
from lerobot.common.OpenC3_robots.src.openc3_mujoco_env import OpenC3PushTEnv

def test_live_ik():
    print("=== 启动 OpenC3 实时运动学逆解 (DLS-IK) 动态跟踪沙盒 ===")
    env = OpenC3PushTEnv(control_freq=30.0)
    obs = env.reset(random_tee=False)

    center = np.array([0.38, 0.0, 0.415], dtype=np.float32)
    radius = 0.06  # 6cm 半径画圆

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for t in range(300):  # 运行 10 秒
            if not viewer.is_running():
                break

            theta = 2.0 * np.pi * (t / 90.0)  # 3 秒转一圈
            target_pos = center + np.array([radius * np.cos(theta), radius * np.sin(theta), 0.0], dtype=np.float32)

            # 1. 逆解求关节角
            q_arm = env.ik_solve_position(target_pos)
            action_7d = np.concatenate([q_arm, [0.0]], dtype=np.float32)

            # 2. 仿真步进并刷新
            obs = env.step(action_7d)
            viewer.sync()

            # 3. 打印跟踪残差
            curr_pusher = env.get_pusher_pos()
            err_mm = np.linalg.norm(curr_pusher - target_pos) * 1000.0
            if t % 15 == 0:
                print(f"[Step {t:03d}] 目标: ({target_pos[0]:.3f}, {target_pos[1]:.3f}) | 实际: ({curr_pusher[0]:.3f}, {curr_pusher[1]:.3f}) | 跟踪误差: {err_mm:.2f} mm")

            time.sleep(1.0 / 30.0)

if __name__ == "__main__":
    test_live_ik()