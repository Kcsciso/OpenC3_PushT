import os
import mujoco
import mujoco.viewer
import numpy as np

os.environ["DISPLAY"] = ":1"
os.environ["MUJOCO_GL"] = "glfw"

XML_PATH = "/home/kasm-user/LLM/lerobot/src/lerobot/common/OpenC3_robots/assets/openc3_scene.xml"
model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# 1. 采用你在图片中调出的最优基准姿态 (微调 J4/J5 消除侧倾)
# J1: 偏向方块, J2: 大臂前探, J3: 肘部弯曲, J4: 小臂微旋, J5: 垂直低头, J6: 归零
q_safe = np.array([-0.12, 0.46, 2.08, -1.05, -1.57, 0.0], dtype=np.float64)

joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}") for i in range(1, 7)]
actuator_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_j{i}") for i in range(1, 7)]
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pusher_site")
tee_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tee_joint")

# 2. 物理清空，速度归零
mujoco.mj_resetData(model, data)
data.qvel[:] = 0.0

# 3. 捡回 T-block 放回桌面标准位置
tee_addr = model.jnt_qposadr[tee_joint_id]
data.qpos[tee_addr : tee_addr + 3] = [0.35, -0.05, 0.4155]
data.qpos[tee_addr + 3 : tee_addr + 7] = [1.0, 0.0, 0.0, 0.0]

# 4. 设置机械臂关节并死死锁住电机
for i, j_id in enumerate(joint_ids):
    data.qpos[model.jnt_qposadr[j_id]] = q_safe[i]
for i, act_id in enumerate(actuator_ids):
    data.ctrl[act_id] = q_safe[i]

mujoco.mj_forward(model, data)

# 5. 计算推杆真实朝向与垂直度的数学夹角
pusher_dir = -data.site_xmat[site_id].reshape(3, 3)[:, 0]
dot = np.clip(np.dot(pusher_dir, [0.0, 0.0, -1.0]), -1.0, 1.0)
tilt_deg = np.rad2deg(np.arccos(dot))
pos = data.site_xpos[site_id]

print("=" * 50)
print(f"✅ T-block 已复位回桌面: [0.35, -0.05, 0.4155]")
print(f"✅ 推杆末端当前坐标: X={pos[0]:.4f}, Y={pos[1]:.4f}, Z={pos[2]:.4f}")
print(f"✅ 推杆数学垂直倾角: {tilt_deg:.2f}° (在 2° 以内即为完全可用状态)")
print("=" * 50)
print(f"确认无误后用于环境的 home_qpos:\nnp.array({list(np.round(q_safe, 4))}, dtype=np.float32)")

# 启动窗口，一切恢复平静
mujoco.viewer.launch(model, data)