
# OpenC3_PushT: 3D Robotic Push-T via LeRobot & Diffusion Policy

基于 **Hugging Face LeRobot** 框架与 **MuJoCo** 物理引擎构建的 3D 机械臂推运基准项目。本项目将经典的 2D Push-T 任务升维至真实物理仿真空间，以开源国产机械臂 **OpenC3 (Collrobstd2)** 为载体，实现了从**场景动力学建模、双目视觉感知、FSM 专家示教、数据采集到 7 维 Diffusion Policy 闭环控制**的完整具身智能流水线。

---

## 核心特性

* **3D 物理动力学仿真**：基于 MuJoCo 深度定制接触刚度、切向摩擦与阻尼参数，采用平滑胶囊体末端推杆，规避推运中的接触穿模与力学奇异性。
* **7-DoF 动作与状态空间**：映射 OpenC3 6 自由度关节 + 1 虚拟工具状态，与 LeRobot 标准 PolicyFeature 规范无缝对齐。
* **多模态双目感知**：同步渲染顶部全局视野（`top_cam`，224×224）与腕部手眼视野（`wrist_cam`，224×224），提供丰富的局部接触与全局位姿感知。
* **抗失稳专家策略 (FSM)**：内置双肩纠偏、质心推运与终态微调的有限状态机专家，支持待机点微扰注入（Noise Injection）以增强鲁棒性。
* **Diffusion Policy 强化训练**：基于预训练 ResNet-18 骨干提取视觉表征，支持时序历史（$T_{\text{obs}}=4$）与动作预测（$T_{\text{pred}}=16$），有效打破非马尔可夫接触混淆。

---

## 项目结构

```text
OpenC3_PushT/
├── assets/
│   ├── openc3_scene.xml          # MuJoCo 物理场景定义（工作台、T 块、机械臂、相机与执行器）
│   └── openc3_real/              # OpenC3 CAD 网格模型 (STL) 与 URDF 定义
├── src/
│   ├── openc3_mujoco_env.py      # MuJoCo Gym-like 仿真环境封装与渲染引擎
│   ├── openc3_robot.py           # 机械臂运动学求解与控制器抽象
│   └── openc3_domain_randomization.py # 纹理、光照与物理域随机化工具
├── tests/
│   ├── calibrate_home_pose.py    # 机械臂零位基准姿态与垂直度标定
│   ├── test_openc3_pusht_env.py  # 仿真环境驱动与基础推运可行性单测
│   ├── demo_openc3_pusht_long.py # FSM 专家长程推运演示与早停机制测试
│   ├── check_openc3_pusht_dataset.py # 标量数据与 MP4 视频流对齐校验
│   └── test_openc3_pusht_closed_loop.py # 训练模型 3D 视窗闭环推运评测
├── record_openc3_pusht_dataset.py# 专家数据集自动录制流水线 (Parquet + H.264)
├── train_openc3_diffusion.py     # 7D Diffusion Policy 训练主入口
├── dev_log.md                    # 架构迭代、踩坑排错与开发日志
└── .gitignore                    # 排除大权重、缓存与高容量数据集

```

---

## 状态与动作空间定义

### 1. 观测空间 (Observation Space)

| 键名 | 类型 | 维度 / 分辨率 | 描述 |
| --- | --- | --- | --- |
| `observation.state`<br> | 状态标量 | `(7,)`<br> | 机械臂 6 个主动关节弧度 + 1 虚拟工具轴状态

 |
| `observation.images.top`<br> | 视觉张量 | `(3, 224, 224)`<br> | 工作台正上方全局俯视视角 (RGB)

 |
| `observation.images.wrist`<br> | 视觉张量 | `(3, 224, 224)`<br> | 末端手眼相机对准推杆切面的局部视角 (RGB)

 |

### 2. 动作空间 (Action Space)

| 键名 | 类型 | 维度 | 描述 |
| --- | --- | --- | --- |
| `action`<br> | 连续控制 | `(7,)`<br> | 机械臂 6 轴期望位置目标角 + 虚拟滑轨位置（归一化至 $[-1, 1]$）

 |

---

## 快速开始

### 1. 环境依赖安装

推荐在 Ubuntu 20.04/22.04、Python 3.10+ 环境下运行：

```bash
# 1. 克隆本仓库到本地（或置于 lerobot 模块下）
git clone [https://github.com/Kcsciso/OpenC3_PushT.git](https://github.com/Kcsciso/OpenC3_PushT.git)
cd OpenC3_PushT

# 2. 安装核心依赖
pip install torch torchvision
pip install mujoco pyav pandas pyarrow tqdm
pip install -e /path/to/lerobot  # 安装本地 LeRobot 库

```

### 2. 环境与基准姿态自检

运行环境单测，确认 IK 求解器与 MuJoCo 物理推运接触正常：

```bash
python tests/test_openc3_pusht_env.py

```

若需可视化查看标定基准姿态（垂直度偏差 $<2^\circ$）：

```bash
python tests/calibrate_home_pose.py

```

### 3. 专家数据采集

运行 FSM 自动数据录制脚本，该脚本会自动对物块生成随机位姿，并在达到高精标准（误差 $<5\text{mm}$ 且偏角 $<6^\circ$ 锁定 30 步）时落盘：

```bash
python record_openc3_pusht_dataset.py

```

* 数据将保存于 `data/openc3_pusht_expert/` 下，包含 `data.parquet` 与 `videos/*.mp4`。



采集完成后，执行结构完整性检查：

```bash
python tests/check_openc3_pusht_dataset.py

```

### 4. 训练 7D Diffusion Policy

启动基于 LeRobot 的扩散模型训练：

```bash
python train_openc3_diffusion.py

```

* 脚本自动计算标量均值/极值并保存为 `dataset_stats.pt`。


* 训练权重与模型配置输出至 `outputs/openc3_pusht_diffusion/final_model/`。



### 5. 策略闭环评测

在 3D 视窗中评测策略在随机初始位姿下的实际推运成功率：

```bash
python tests/test_openc3_pusht_closed_loop.py

```

---

## 达标收敛基准 (Metrics)

推运任务以工业装配公差为基准判据：

* **位移误差**：$\Delta d = \|\mathbf{p}_{\text{tee}} - \mathbf{p}_{\text{goal}}\| < 5.0\text{ mm}$

* **姿态误差**：$|\text{Yaw}_{\text{err}}| < 6.0^\circ$

* **稳定锁定**：连续维持在公差带内超 30 步（$1.0\text{ s}$ @ 30 FPS）判定为成功回合。



```
