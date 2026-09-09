import os
os.environ["DISPLAY"] = ":1"
os.environ["MUJOCO_GL"] = "glfw"

import shutil
import time
from pathlib import Path
import av
import numpy as np
import pandas as pd
import torch
import mujoco
import mujoco.viewer

from lerobot.common.OpenC3_robots.src.openc3_mujoco_env import OpenC3PushTEnv

DATASET_DIR = Path(__file__).resolve().parent / "data" / "openc3_pusht_expert"
FPS = 30
EPISODE_LENGTH = int(50.0 * FPS)  # 50 秒最大防超时容限 (1500 帧)
NUM_EPISODES = 100                 # 目标达标回合数
SUCCESS_DIST_MM = 5.0
SUCCESS_YAW_DEG = 6.0


class PushTExpert:
    """双肩差动抗失稳推进专家策略"""
    def __init__(self, env: OpenC3PushTEnv):
        self.env = env
        self.goal_pos = np.array([0.38, 0.0], dtype=np.float32)
        self.z_hover = 0.485
        self.z_push = 0.412

        self.state = "HOVER"
        self.state_step = 0
        self.standoff_xy = None
        self.push_target_xy = None
        self.hover_start_xy = None
        self.lift_origin_xy = None
        self.is_success_locked = False

    def reset_fsm(self):
        self.state = "HOVER"
        self.state_step = 0
        self.standoff_xy = None
        self.push_target_xy = None
        self.hover_start_xy = None
        self.lift_origin_xy = None
        self.is_success_locked = False

    def get_action_step(self, tee_pos):
        _, curr_quat = self.env.get_tee_pose()
        w, x, y, z = curr_quat
        curr_yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        dist_to_goal = float(np.linalg.norm(tee_pos[:2] - self.goal_pos))
        yaw_err = curr_yaw

        # 高精收敛判定 (< 4.2mm 锁定，> 4.8mm 解锁补推)
        if dist_to_goal < 0.0042 and abs(yaw_err) < 0.087:
            self.is_success_locked = True
        elif dist_to_goal > 0.0048 or abs(yaw_err) > 0.105:
            self.is_success_locked = False

        curr_pusher = self.env.get_pusher_pos()
        if self.is_success_locked:
            return np.array([curr_pusher[0], curr_pusher[1], self.z_hover], dtype=np.float32)

        steps_hover = 10
        steps_descend = 15
        steps_push = 35
        steps_lift = 10

        if self.state == "HOVER":
            if self.state_step == 0:
                self.hover_start_xy = curr_pusher[:2].copy()
                cos_y, sin_y = np.cos(curr_yaw), np.sin(curr_yaw)
                R = np.array([[cos_y, -sin_y], [sin_y, cos_y]], dtype=np.float32)

                world_vec = self.goal_pos - tee_pos[:2]
                local_vx = float(cos_y * world_vec[0] + sin_y * world_vec[1])
                local_vy = float(-sin_y * world_vec[0] + cos_y * world_vec[1])

                r_pusher = 0.008
                clearance = 0.008
                nominal_push = float(np.clip(dist_to_goal * 0.70 + 0.005, 0.006, 0.022))

                # ---------------- 策略 1: 终态微调 ----------------
                if dist_to_goal <= 0.014 and abs(yaw_err) > 0.07:
                    effective_push = float(np.clip(abs(yaw_err) * 0.035, 0.003, 0.006))
                    x_contact = 0.035 if yaw_err > 0 else -0.035
                    face_pt = np.array([x_contact, 0.060], dtype=np.float32)
                    face_n  = np.array([0.0, 1.0], dtype=np.float32)

                # ---------------- 策略 2: 纵向推进 (双肩纠偏 + 中心对齐) ----------------
                elif abs(local_vy) >= abs(local_vx):
                    if local_vy > 0:
                        face_n = np.array([0.0, -1.0], dtype=np.float32)
                        if yaw_err < -0.035:
                            # 顺时针偏：推右肩正下方 (X = +0.032)，留出 9mm 侧向净空
                            effective_push = float(np.clip(nominal_push, 0.008, 0.018))
                            face_pt = np.array([0.032, 0.030], dtype=np.float32)
                        elif yaw_err > 0.035:
                            # 逆时针偏：推左肩正下方 (X = -0.032)
                            effective_push = float(np.clip(nominal_push, 0.008, 0.018))
                            face_pt = np.array([-0.032, 0.030], dtype=np.float32)
                        else:
                            # 姿态摆正：严格推竖梁底部中心 (X = 0.0) 对称推进
                            effective_push = float(np.clip(nominal_push, 0.008, 0.020))
                            face_pt = np.array([0.0, -0.060], dtype=np.float32)
                    else:
                        effective_push = float(np.clip(nominal_push, 0.008, 0.022))
                        x_corr = float(np.clip(yaw_err * 0.08, -0.025, 0.025))
                        face_pt = np.array([x_corr, 0.060], dtype=np.float32)
                        face_n  = np.array([0.0, 1.0], dtype=np.float32)

                # ---------------- 策略 3: 横向推进 (核心修复：对齐质心 Y=0.012) ----------------
                else:
                    move_right = local_vx > 0
                    effective_push = float(np.clip(nominal_push, 0.008, 0.020))
                    # 关键修改：质心在 Y=+0.018，接触点锚定在 Y=+0.012，推力直穿质心且留有 18mm 肩部净空
                    # 横向推进：严格避开 Y in [0.01, 0.04] 的横梁内凹角
                    y_safe = -0.020  # 距离横梁肩膀整整 50mm 净空，推杆绝不会刮到横梁
                    if move_right:
                        face_pt = np.array([-0.015, y_safe], dtype=np.float32)
                        face_n  = np.array([-1.0, 0.0], dtype=np.float32)
                    else:
                        face_pt = np.array([0.015, y_safe], dtype=np.float32)
                        face_n  = np.array([1.0, 0.0], dtype=np.float32)

                standoff_pt = face_pt + face_n * (r_pusher + clearance)
                target_pt   = face_pt + face_n * (r_pusher - effective_push)

                # =============================================================
                # 【新增】仅给落刀起始点注入 ±2mm 空间扰动，模拟视觉推理时的落刀误差
                # =============================================================
                noise_xy = np.random.uniform(-0.002, 0.002, size=2).astype(np.float32)
                
                # 落刀待机点带噪声，推杆会稍微偏落 1~2mm
                self.standoff_xy = tee_pos[:2] + R @ standoff_pt + noise_xy
                # 推运目标点不加噪声，依然稳稳推向真实目标，训练策略自我纠偏
                self.push_target_xy = tee_pos[:2] + R @ target_pt

            alpha = min(1.0, self.state_step / float(steps_hover))
            target_xy = (1.0 - alpha) * self.hover_start_xy + alpha * self.standoff_xy
            target_z = self.z_hover

        elif self.state == "DESCEND":
            alpha = min(1.0, self.state_step / float(steps_descend))
            target_xy = self.standoff_xy
            target_z = (1.0 - alpha) * self.z_hover + alpha * self.z_push

        elif self.state == "PUSH":
            alpha = min(1.0, self.state_step / float(steps_push))
            target_xy = (1.0 - alpha) * self.standoff_xy + alpha * self.push_target_xy
            target_z = self.z_push

        elif self.state == "LIFT":
            alpha = min(1.0, self.state_step / float(steps_lift))
            target_xy = self.lift_origin_xy
            target_z = (1.0 - alpha) * self.z_push + alpha * self.z_hover

        else:
            target_xy = curr_pusher[:2]
            target_z = self.z_hover

        self.state_step += 1
        if self.state == "HOVER" and self.state_step >= steps_hover:
            self.state = "DESCEND"
            self.state_step = 0
        elif self.state == "DESCEND" and self.state_step >= steps_descend:
            self.state = "PUSH"
            self.state_step = 0
        elif self.state == "PUSH" and self.state_step >= steps_push:
            self.lift_origin_xy = curr_pusher[:2].copy()
            self.state = "LIFT"
            self.state_step = 0
        elif self.state == "LIFT" and self.state_step >= steps_lift:
            self.state = "HOVER"
            self.state_step = 0

        return np.array([target_xy[0], target_xy[1], target_z], dtype=np.float32)


def encode_video(frames: list, output_path: Path, fps: int = 30):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(output_path), mode="w")
    stream = container.add_stream("h264", rate=fps)
    stream.width = frames[0].shape[1]
    stream.height = frames[0].shape[0]
    stream.pix_fmt = "yuv420p"

    for frame in frames:
        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for packet in stream.encode(av_frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def record_dataset():
    if DATASET_DIR.exists():
        print(f"清理旧专家数据集: {DATASET_DIR}")
        shutil.rmtree(DATASET_DIR)

    videos_dir = DATASET_DIR / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    env = OpenC3PushTEnv(control_freq=FPS)
    expert = PushTExpert(env)

    all_states = []
    all_actions = []
    episode_indices = []
    frame_indices = []

    print(f"=== 启动 Open_C3 Push-T 专家数据采集 (目标: {NUM_EPISODES} 组达标数据 | 上限: 50.0s/集) ===")

    # 启动 3D 交互视窗，支持全程观察机械臂实时推运过程
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        recorded_episodes = 0
        attempt_count = 0

        while recorded_episodes < NUM_EPISODES:
            if not viewer.is_running():
                print("\n3D 视窗已被手动关闭，数据录制终止。")
                break

            attempt_count += 1
            obs = env.reset(random_tee=True)
            expert.reset_fsm()
            init_tee_pos, _ = env.get_tee_pose()
            init_dist_mm = float(np.linalg.norm(init_tee_pos[:2] - expert.goal_pos) * 1000.0)

            ep_states = []
            ep_actions = []
            top_frames = []
            wrist_frames = []
            success_hold_steps = 0

            for step in range(EPISODE_LENGTH):
                if not viewer.is_running():
                    break

                step_start = time.perf_counter()

                curr_tee_pos, _ = env.get_tee_pose()
                target_cartesian = expert.get_action_step(curr_tee_pos)

                # 达标早停机制：稳定悬停 30 步 (1.0 秒) 立即截停本集，转入下一集
                if expert.is_success_locked:
                    success_hold_steps += 1
                    if success_hold_steps >= 30:
                        break
                else:
                    success_hold_steps = 0

                q_arm = env.ik_solve_position(target_cartesian)
                action_7d = np.concatenate([q_arm, [0.0]], dtype=np.float32)

                state = obs["observation.state"].numpy()
                top_img = (obs["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                wrist_img = (obs["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                ep_states.append(state)
                ep_actions.append(action_7d)
                top_frames.append(top_img)
                wrist_frames.append(wrist_img)

                # 步进物理仿真并刷新桌面视窗画面
                obs = env.step(action_7d)
                viewer.sync()

                # 维持真实流速，便于人眼在桌面端监视仿真推运
                elapsed = time.perf_counter() - step_start
                if elapsed < 1.0 / FPS:
                    time.sleep(1.0 / FPS - elapsed)

            # 回合指标结算
            final_tee_pos, final_quat = env.get_tee_pose()
            w, x, y, z = final_quat
            final_yaw_deg = float(np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))))
            final_dist_mm = float(np.linalg.norm(final_tee_pos[:2] - expert.goal_pos) * 1000.0)
            ep_frames = len(top_frames)
            is_succ = (final_dist_mm < SUCCESS_DIST_MM) and (abs(final_yaw_deg) < SUCCESS_YAW_DEG)

            # 每集状态即时报告
            if is_succ:
                for f_idx in range(ep_frames):
                    all_states.append(ep_states[f_idx])
                    all_actions.append(ep_actions[f_idx])
                    episode_indices.append(recorded_episodes)
                    frame_indices.append(f_idx)

                encode_video(top_frames, videos_dir / f"observation.images.top_episode_{recorded_episodes:06d}.mp4", fps=FPS)
                encode_video(wrist_frames, videos_dir / f"observation.images.wrist_episode_{recorded_episodes:06d}.mp4", fps=FPS)

                recorded_episodes += 1
                print(
                    f"[Episode {recorded_episodes:02d}/{NUM_EPISODES}] (第 {attempt_count} 次试验) | "
                    f"耗时: {ep_frames/FPS:4.1f}s ({ep_frames:3d} 帧) | "
                    f"初距: {init_dist_mm:4.1f} mm -> 终距: {final_dist_mm:3.1f} mm | "
                    f"偏角: {final_yaw_deg:4.1f}° | ✅ 正常达标 (Saved)",
                    flush=True,
                )
            else:
                print(
                    f"[Episode --/{NUM_EPISODES}] (第 {attempt_count} 次试验) | "
                    f"耗时: {ep_frames/FPS:4.1f}s ({ep_frames:3d} 帧) | "
                    f"初距: {init_dist_mm:4.1f} mm -> 终距: {final_dist_mm:3.1f} mm | "
                    f"偏角: {final_yaw_deg:4.1f}° | ⚠️ 异常未达标 (已丢弃重试)",
                    flush=True,
                )

    df = pd.DataFrame(
        {
            "episode_index": episode_indices,
            "frame_index": frame_indices,
            "index": list(range(len(all_states))),
            "observation.state": [s.tolist() for s in all_states],
            "action": [a.tolist() for a in all_actions],
        }
    )
    df.to_parquet(DATASET_DIR / "data.parquet", index=False)
    print("\n" + "=" * 60)
    print(f"数据采集完成！共记录 {recorded_episodes} 组高质量达标 Episode，有效总帧数: {len(all_states)}")
    print(f"数据集落盘位置: {DATASET_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    record_dataset()