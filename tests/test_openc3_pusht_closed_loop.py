import os
os.environ["DISPLAY"] = ":1"
os.environ["MUJOCO_GL"] = "glfw"

import time
from pathlib import Path
import mujoco
import mujoco.viewer
import numpy as np
import pandas as pd
import torch

from lerobot.common.OpenC3_robots.src.openc3_mujoco_env import OpenC3PushTEnv
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent.parent if (Path(__file__).resolve().parent.parent / "data").exists() else Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "outputs" / "openc3_pusht_diffusion" / "final_model"
DATASET_DIR = BASE_DIR / "data" / "openc3_pusht_expert"


def load_dataset_stats() -> dict:
    candidate_paths = [
        MODEL_DIR / "dataset_stats.pt",
        MODEL_DIR / "stats.pt",
        DATASET_DIR / "dataset_stats.pt",
        DATASET_DIR / "stats.pt",
    ]

    for path in candidate_paths:
        if path.exists():
            stats = torch.load(path, map_location=DEVICE)
            for k in stats:
                for sub_k in stats[k]:
                    if isinstance(stats[k][sub_k], torch.Tensor):
                        stats[k][sub_k] = stats[k][sub_k].to(DEVICE)
                    else:
                        stats[k][sub_k] = torch.tensor(stats[k][sub_k], device=DEVICE, dtype=torch.float32)
            return stats

    raise FileNotFoundError(f"未找到统计量文件，请先运行训练脚本！")


def unnormalize_action(norm_action: torch.Tensor, stats: dict) -> np.ndarray:
    norm_action_clamped = torch.clamp(norm_action, -1.0, 1.0)
    a_min = stats["action"]["min"]
    a_max = stats["action"]["max"]
    raw_action = (norm_action_clamped + 1.0) / 2.0 * (a_max - a_min) + a_min
    return raw_action.detach().cpu().numpy()


def quat_to_yaw(quat: np.ndarray) -> float:
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


# 将单轮最大步数由 600 改为 1500 步 (50.0 秒 @ 30 FPS)
def run_single_rollout(env, policy, stats, viewer, max_steps: int = 1500, random_tee: bool = True):
    obs = env.reset(random_tee=random_tee)
    policy.reset()

    goal_pos = np.array([0.38, 0.0], dtype=np.float32)
    init_pos, _ = env.get_tee_pose()
    init_dist_mm = np.linalg.norm(init_pos[:2] - goal_pos) * 1000.0

    success_hold_steps = 0
    is_success = False

    for step in range(max_steps):
        step_start = time.perf_counter()

        curr_state = obs["observation.state"].to(DEVICE)
        s_min = stats["observation.state"]["min"]
        s_max = stats["observation.state"]["max"]
        diff = s_max - s_min
        scale = torch.where(diff < 1e-4, torch.ones_like(diff), diff)
        norm_state = torch.clamp(2.0 * (curr_state - s_min) / scale - 1.0, -1.0, 1.0)

        # 单帧输入，时序由 LeRobot 内部 FIFO 队列接管
        batch = {
            "observation.state": norm_state.unsqueeze(0),
            "observation.images.top": obs["observation.images.top"].unsqueeze(0).to(DEVICE),
            "observation.images.wrist": obs["observation.images.wrist"].unsqueeze(0).to(DEVICE),
        }

        with torch.no_grad():
            norm_action = policy.select_action(batch)

        action_7d = unnormalize_action(norm_action.squeeze(0), stats)
        obs = env.step(action_7d)
        if viewer:
            viewer.sync()

        tee_pos, tee_quat = env.get_tee_pose()
        curr_dist_mm = np.linalg.norm(tee_pos[:2] - goal_pos) * 1000.0
        curr_yaw_deg = np.degrees(quat_to_yaw(tee_quat))

        # 连续 30 步 (1.0 秒) 维持工业精度，立即判赢早停
        if curr_dist_mm < 5.0 and abs(curr_yaw_deg) < 6.0:
            success_hold_steps += 1
            if success_hold_steps >= 30:
                is_success = True
                break
        else:
            success_hold_steps = 0

        elapsed = time.perf_counter() - step_start
        if elapsed < 1.0 / 30.0:
            time.sleep(1.0 / 30.0 - elapsed)

    final_pos, final_quat = env.get_tee_pose()
    final_dist_mm = np.linalg.norm(final_pos[:2] - goal_pos) * 1000.0
    final_yaw_deg = np.degrees(quat_to_yaw(final_quat))

    return is_success, init_dist_mm, final_dist_mm, final_yaw_deg, step + 1


def evaluate_closed_loop(num_episodes: int = 10):
    print(f"=== [策略评测] 运行 Open_C3 Push-T 闭环测试 (共 {num_episodes} 轮 | 上限 50.0 秒) ===")
    assert MODEL_DIR.exists(), f"未找到已训练权重: {MODEL_DIR}"

    policy = DiffusionPolicy.from_pretrained(MODEL_DIR)
    policy.to(DEVICE)
    policy.eval()
    stats = load_dataset_stats()

    env = OpenC3PushTEnv(control_freq=30.0)
    success_count = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for ep in range(num_episodes):
            if not viewer.is_running():
                break

            succ, i_dist, f_dist, f_yaw, steps = run_single_rollout(
                env, policy, stats, viewer, max_steps=1500, random_tee=True
            )
            if succ:
                success_count += 1

            status = "✅ SUCCESS" if succ else "❌ FAILED"
            print(
                f"Episode {ep+1:02d}/{num_episodes} | 耗时: {steps/30.0:4.1f}s ({steps:4d}步) | "
                f"初距: {i_dist:4.1f} mm -> 终距: {f_dist:3.1f} mm | 偏角: {f_yaw:4.1f}° | {status}"
            )

    success_rate = (success_count / num_episodes) * 100.0
    print("\n" + "=" * 50)
    print(f"评测汇总 | 达标率: {success_count}/{num_episodes} ({success_rate:.1f}%)")
    print("=" * 50)


if __name__ == "__main__":
    evaluate_closed_loop(num_episodes=10)