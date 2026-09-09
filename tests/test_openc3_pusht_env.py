import os
import sys

if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"

import time
import numpy as np
import torch
from lerobot.common.OpenC3_robots.src.openc3_mujoco_env import OpenC3PushTEnv


def test_pusht_environment():
    print("=== [步骤 1 校验] 正在测试 Open_C3 3D Push-T 物理仿真环境 ===")
    env = OpenC3PushTEnv(control_freq=30.0)
    obs = env.reset(random_tee=False)

    init_tee_pos, _ = env.get_tee_pose()
    init_pusher_pos = env.get_pusher_pos()
    print(f"推杆初始坐标: X={init_pusher_pos[0]:.3f}, Y={init_pusher_pos[1]:.3f}, Z={init_pusher_pos[2]:.3f}")
    print(f"T-Block 初始坐标: X={init_tee_pos[0]:.3f}, Y={init_tee_pos[1]:.3f}, Z={init_tee_pos[2]:.3f}")

    print("\n执行 60 步笛卡尔路径推运测试 (推杆高度保持 Z=0.415m 贴地推进)...")

    # 规划推杆路径: 从 T 块后方 (0.32, -0.16, 0.415) 推进至 (0.32, -0.02, 0.415)
    for step in range(60):
        target_y = -0.16 + (step / 60.0) * 0.14
        cartesian_target = np.array([0.32, target_y, 0.415], dtype=np.float32)

        q_arm = env.ik_solve_position(cartesian_target)
        action_7d = np.concatenate([q_arm, [0.0]], dtype=np.float32)
        obs = env.step(action_7d)

    final_tee_pos, _ = env.get_tee_pose()
    final_pusher_pos = env.get_pusher_pos()
    distance_moved = np.linalg.norm(final_tee_pos[:2] - init_tee_pos[:2])

    print(f"\n推杆结束坐标: X={final_pusher_pos[0]:.3f}, Y={final_pusher_pos[1]:.3f}, Z={final_pusher_pos[2]:.3f}")
    print(f"T-Block 结束坐标: X={final_tee_pos[0]:.3f}, Y={final_tee_pos[1]:.3f}, Z={final_tee_pos[2]:.3f}")
    print(f"T-Block 物理推运位移: {distance_moved * 1000:.1f} mm")

    # 校验
    assert obs["observation.state"].shape == (7,)
    assert obs["observation.images.top"].shape == (3, 224, 224)
    assert obs["observation.images.wrist"].shape == (3, 224, 224)
    assert distance_moved > 0.015, f"T 块位移不足 ({distance_moved*1000:.1f}mm)，推运未达标！"

    print("\n✅ Open_C3 Push-T 物理环境通过验证，末端 IK 驱动与刚体接触推运完全生效！")


if __name__ == "__main__":
    test_pusht_environment()