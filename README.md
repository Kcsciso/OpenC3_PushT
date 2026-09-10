# OpenC3_PushT: 3D Robotic Push-T via LeRobot & Diffusion Policy

基于 **Hugging Face LeRobot** 框架与 **MuJoCo** 物理引擎构建的 3D 机械臂推运基准项目[cite: 1, 2]。本项目将经典的 2D Push-T 任务升维至真实物理仿真空间，以开源国产机械臂 **OpenC3 (Collrobstd2)** 为载体，实现了从**场景动力学建模、双目视觉感知、FSM 专家示教、数据采集到 7 维 Diffusion Policy 闭环控制**的完整具身智能流水线[cite: 1, 2]。

---

## 核心特性

* **3D 物理动力学仿真**：基于 MuJoCo 深度定制接触刚度、切向摩擦与阻尼参数，采用平滑胶囊体末端推杆，规避推运中的接触穿模与力学奇异性[cite: 1, 2]。
* **7-DoF 动作与状态空间**：映射 OpenC3 6 自由度关节 + 1 虚拟工具状态，与 LeRobot 标准 PolicyFeature 规范无缝对齐[cite: 1, 2]。
* **多模态双目感知**：同步渲染顶部全局视野（`top_cam`，224×224）与腕部手眼视野（`wrist_cam`，224×224），提供丰富的局部接触与全局位姿感知[cite: 1, 2]。
* **抗失稳专家策略 (FSM)**：内置双肩纠偏、质心推运与终态微调的有限状态机专家，支持待机点微扰注入（Noise Injection）以增强鲁棒性[cite: 1, 2]。
* **Diffusion Policy 强化训练**：基于预训练 ResNet-18 骨干提取视觉表征，支持时序历史（$T_{\text{obs}}=4$）与动作预测（$T_{\text{pred}}=16$），有效打破非马尔可夫接触混淆[cite: 1, 2]。

---

## 全链路技术架构（四阶段演进体系）

本项目包含具身智能控制全生命周期的四个核心研发阶段[cite: 1, 2]：

### 阶段一：物理建模与 DLS-IK 运动学解算
* **动力学与接触建模**：在 `assets/openc3_scene.xml` 中将推杆末端重构为半球底胶囊体（半径 $8\text{ mm}$），彻底消除圆柱边缘法向量阶跃跳变引起的击飞现象[cite: 1, 2]；推运接触面设置临界阻尼（`solref="0.008 1.0"`）与渐进阻抗（`solimp="0.8 0.95 0.001"`），压制高频颤动[cite: 1, 2]。
* **DLS 阻尼最小二乘逆运动学**：在 `src/openc3_mujoco_env.py` 中释放推杆绕自身中心轴自转的偏航自由度，将 6D 姿态约束降维为 3D 位置跟踪；在奇异点附近引入动态阻尼矩阵 $\lambda^2 I$，避免关节角加速度发散[cite: 1, 2]。
* **零空间投影姿态吸附**：利用 $6 - 3 = 3$ 个多余冗余自由度，构建零空间投影算子 $(I - J^\dagger J)$，实时将机械臂拉向标定好的最优支撑构型 `home_qpos`，杜绝大臂向外翻折打在桌面[cite: 1, 2]。

### 阶段二：抗失稳 FSM 专家策略与力矩平衡
* **世界坐标系投影解耦**：通过方块当前 Yaw 角旋转矩阵反解局部移动矢量，打破沿局部轴线推进导致的“倒立摆横向漂移”[cite: 1, 2]。
* **双肩差动纠偏（Shoulder Differential Push）**：偏角过大时放弃推立柱底端，转而推进横梁两翼下沿肩膀（$X = \pm 0.032\text{ m}, Y = +0.030\text{ m}$）[cite: 1, 2]；既保留 9 mm 侧向绝对净空避免直角卡死，又利用偏心力矩实现“边向前推进边把姿态扳正”[cite: 1, 2]。
* **抗协变量偏移噪声注入（Sim2Real 预备）**：在落刀待机点注入 $\pm 2\text{ mm}$ 空间均匀随机噪声，目标推运点保持精确，使策略在模仿学习中掌握误差自纠偏能力[cite: 1, 2]。

### 阶段三：轻量化多模态数据流与质量工程
* **MP4 (H.264) + Parquet 混合存储架构**：使用 `PyAV` 将两路 224×224 RGB 视觉流硬件压缩为 MP4 视频，控制标量采用 Apache Arrow Parquet 列式存储，将数据吞吐体积降低数十倍并消除磁盘碎文件 I/O 阻塞[cite: 1, 2]。
* **原子级早停与拒止采样**：引入连续 30 步（$1.0\text{ s}$ @ 30 FPS）工业精度达标即刻截断机制，剔除静止死锁帧[cite: 1, 2]；未达标轮次（$>5.0\text{ mm}$ 或 $>6.0^\circ$）直接丢弃重试，确保全量数据无瑕疵[cite: 1, 2]。

### 阶段四：Diffusion Policy 与动作分块闭环
* **时序感知与去噪生成**：输入 4 帧历史观测（$T_{\text{obs}}=4$，0.133 秒时序上下文）打破接触几何高度二义性[cite: 1, 2]；采用预训练 ResNet-18 提取图像特征，克服 MSE 策略的均值崩溃现象[cite: 1, 2]。
* **动作分块（Action Chunking）**：预测视界 $T_{\text{pred}}=16$，执行步长 $n_{\text{action\_steps}}=12$[cite: 1, 2]；利用动作块本身的物理动量保持推运流畅性，避免高频打断与动作抽搐[cite: 1, 2]。
* **LeRobot FIFO 队列透明接管**：闭环测试端以单帧方式调用，内部维护观测与动作滑窗，兼顾低计算延迟与滚动纠偏[cite: 1, 2]。

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

| 键名 | 数据类型 | 维度 / 格式 | 取值范围 / 单位 | 描述 |
| :--- | :--- | :--- | :--- | :--- |
| `observation.state` | Float32 Tensor | `(7,)` | 弧度 (rad) / 米 (m) | 机械臂 6 个主动关节角度 + 1 虚拟工具滑轨状态 |
| `observation.images.top` | Float32 Tensor | `(3, 224, 224)` | $[0.0, 1.0]$ RGB | 工作台全局正顶视俯视角画面 (C, H, W) |
| `observation.images.wrist` | Float32 Tensor | `(3, 224, 224)` | $[0.0, 1.0]$ RGB | 末端手眼相机对准推杆切面的局部视野 (C, H, W) |

### 2. 动作空间 (Action Space)

| 键名 | 数据类型 | 维度 | 物理空间 (Env / Parquet) | 模型空间 (Policy In/Out) | 描述 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `action` | Float32 Tensor | `(7,)` | 目标关节角 (rad) + 滑轨位置 (m) | 线性归一化至 $[-1.0, 1.0]$ | 机械臂 6 轴期望位置目标角 + 虚拟滑轨控制量 |


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

* 脚本自动计算标量极值并保存为 `dataset_stats.pt`。


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

---

### GitHub 代码提交与推送命令

在终端依次运行以下命令，即可将更新后的 `README.md` 推送至远程仓库：

```bash
# 1. 切换到独立仓库根目录
cd /home/kasm-user/LLM/lerobot/src/lerobot/common/OpenC3_robots

# 2. 检查暂存区状态（确认仅更改了 README.md）
git status

# 3. 暂存并提交更新
git add README.md
git commit -m "docs: add four-stage full-stack technical architecture to README"

# 4. 推送到 GitHub 远程 main 分支
git push origin main

```