import sys, os
os.environ["DISPLAY"] = ":1"
os.environ["MUJOCO_GL"] = "glfw"

import time
import mujoco.viewer
import numpy as np
from lerobot.common.OpenC3_robots.src.openc3_mujoco_env import OpenC3PushTEnv


class PushTExpert:
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

        # 严格对齐阈值：< 4.2mm 锁定，> 4.8mm 解锁补推，确保稳稳收敛在 5.0mm 工业达标线内
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

                # 冲程根据当前距目标的远近自适应收缩 (防终点暴冲撞飞)
                nominal_push = float(np.clip(dist_to_goal * 0.70 + 0.005, 0.006, 0.022))

                # -------------------------------------------------------------
                # 策略 1: 终态微调 (已接近目标 <= 14mm 且偏角偏大，轻敲顶面远端)
                # -------------------------------------------------------------
                if dist_to_goal <= 0.014 and abs(yaw_err) > 0.07:
                    effective_push = float(np.clip(abs(yaw_err) * 0.035, 0.003, 0.006))
                    x_contact = 0.035 if yaw_err > 0 else -0.035
                    face_pt = np.array([x_contact, 0.060], dtype=np.float32)
                    face_n  = np.array([0.0, 1.0], dtype=np.float32)

                # -------------------------------------------------------------
                # 策略 2: 纵向推进 (双肩纠偏 + 居中对称正推)
                # -------------------------------------------------------------
                elif abs(local_vy) >= abs(local_vx):
                    if local_vy > 0:
                        # 向上正推 (+Y)
                        face_n = np.array([0.0, -1.0], dtype=np.float32)

                        if yaw_err < -0.035:
                            # 顺时针偏角：推右肩正下方 (X = +0.032)，净空 9mm 绝不刮擦竖梁
                            effective_push = float(np.clip(nominal_push, 0.008, 0.018))
                            face_pt = np.array([0.032, 0.030], dtype=np.float32)
                        elif yaw_err > 0.035:
                            # 逆时针偏角：推左肩正下方 (X = -0.032)，净空 9mm
                            effective_push = float(np.clip(nominal_push, 0.008, 0.018))
                            face_pt = np.array([-0.032, 0.030], dtype=np.float32)
                        else:
                            # 姿态已摆正：严格推竖梁底部正中心 (X=0.0)，消除单侧推运导致的侧滑偏航
                            effective_push = float(np.clip(nominal_push, 0.008, 0.020))
                            face_pt = np.array([0.0, -0.060], dtype=np.float32)

                    else:
                        # 向下推 (-Y)：推横梁顶面
                        effective_push = float(np.clip(nominal_push, 0.008, 0.022))
                        x_corr = float(np.clip(yaw_err * 0.08, -0.025, 0.025))
                        face_pt = np.array([x_corr, 0.060], dtype=np.float32)
                        face_n  = np.array([0.0, 1.0], dtype=np.float32)

                # -------------------------------------------------------------
                # 策略 3: 横向推进 (推竖梁下半段侧壁，Y=-0.035 避开肩部)
                # -------------------------------------------------------------
                else:
                    move_right = local_vx > 0
                    effective_push = float(np.clip(nominal_push, 0.008, 0.020))
                    y_safe = -0.035
                    if move_right:
                        face_pt = np.array([-0.015, y_safe], dtype=np.float32)
                        face_n  = np.array([-1.0, 0.0], dtype=np.float32)
                    else:
                        face_pt = np.array([0.015, y_safe], dtype=np.float32)
                        face_n  = np.array([1.0, 0.0], dtype=np.float32)

                standoff_pt = face_pt + face_n * (r_pusher + clearance)
                target_pt   = face_pt + face_n * (r_pusher - effective_push)

                self.standoff_xy = tee_pos[:2] + R @ standoff_pt
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


def run_long_demo(duration_seconds: float = 50.0):
    duration_seconds = 50.0
    print(f"=== 启动 Open_C3 3D Push-T 长程多轮推运演示 ({duration_seconds:.0f} 秒) ===")
    env = OpenC3PushTEnv(control_freq=30.0)
    expert = PushTExpert(env)

    obs = env.reset(random_tee=True)
    expert.reset_fsm()
    init_pos, _ = env.get_tee_pose()
    print(f"随机起点坐标: ({init_pos[0]:.3f}, {init_pos[1]:.3f}) | 初始距目标: {np.linalg.norm(init_pos[:2] - np.array([0.38, 0.0]))*1000:.1f} mm")

    total_steps = int(duration_seconds * 30.0)
    goal_pos = np.array([0.38, 0.0])
    success_hold_steps = 0

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        for step in range(total_steps):
            if not viewer.is_running():
                break

            step_start = time.perf_counter()

            curr_tee_pos, _ = env.get_tee_pose()
            target_cartesian = expert.get_action_step(curr_tee_pos)

            if expert.is_success_locked:
                success_hold_steps += 1
                if success_hold_steps >= 30:
                    print(f"\n>>> [达标早停] 机械臂已在高精收敛区安全悬停 1.0 秒，演示圆满完成！")
                    break
            else:
                success_hold_steps = 0

            q_arm = env.ik_solve_position(target_cartesian)
            action_7d = np.concatenate([q_arm, [0.0]], dtype=np.float32)

            env.step(action_7d)
            viewer.sync()

            curr_pos, curr_quat = env.get_tee_pose()
            w, x, y, z = curr_quat
            curr_yaw_deg = np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
            curr_dist = np.linalg.norm(curr_pos[:2] - goal_pos) * 1000.0

            if step % 30 == 0:
                print(
                    f"[{step/30.0:4.1f}s / {duration_seconds:.0f}s] 状态: {expert.state:7s} | "
                    f"距目标: {curr_dist:5.1f} mm | 偏角: {curr_yaw_deg:5.1f}° | 达标锁定: {expert.is_success_locked}"
                )

            elapsed = time.perf_counter() - step_start
            if elapsed < 1.0 / 30.0:
                time.sleep(1.0 / 30.0 - elapsed)

        final_pos, final_quat = env.get_tee_pose()
        w, x, y, z = final_quat
        final_yaw_deg = np.degrees(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
        final_dist = np.linalg.norm(final_pos[:2] - goal_pos) * 1000.0
        is_success = final_dist < 5.0 and abs(final_yaw_deg) < 6.0

        if hasattr(env, "viewer") and env.viewer:
            env.viewer.close()

    print("\n" + "=" * 50)
    print(f"演示结束 | 最终距目标: {final_dist:.1f} mm | 姿态偏角: {final_yaw_deg:.1f}°")
    print(f"工业级精度达标判定 (<5.0mm & <6.0°): {'✅ SUCCESS' if is_success else '❌ FAILED'}")
    print("=" * 50)
    time.sleep(1.0)
    os._exit(0)


if __name__ == "__main__":
    run_long_demo(duration_seconds=50.0)