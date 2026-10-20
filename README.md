# 项目结构 / Project Structure

本项目为小组作业 **AY10：基于强化学习的交通信号控制**。技术路线与实验设计见 `plan.md`，开题报告的方案部分见 `proposal/`。

图例：✅ 已存在　🚧 计划中（尚未创建）

```
AY10/
├── README.md                       # 本文件：项目结构与环境配置说明            ✅
├── AY10.pdf                        # 课程作业原始要求                          ✅
├── plan.md                         # 方案与开题报告思路稿（已定技术路线）        ✅
├── proposal/                       # 开题报告（LaTeX 源文件与 PDF）             ✅
│   ├── proposal.tex                #   方案部分正文（XeLaTeX 编译）
│   └── proposal.pdf                #   编译产物
├── scripts/ 
│   ├── check_env.py                # 检查配置环境
│   └── demo_fixed_time.py          # 检查仿真环境
├── requirements.txt                # Python 依赖清单                          🚧
├── nets/                           # SUMO 路网、需求与信号配时                   🚧
│   └── single-intersection/
│       ├── single-intersection.net.xml   # 路网：单路口，4 相位、每方向 3 车道
│       ├── single-intersection.rou.xml   # 需求：各方向泊松到达 + 转向比例
│       ├── tls.add.xml                   # 信号程序：4 绿灯相位 + 黄灯过渡
│       └── single-intersection.sumocfg   # 仿真运行配置
├── src/                            # 源代码                                   🚧
│   ├── env/
│   │   ├── sumo_env.py             #   封装 SumoEnvironment、场景与随机种子
│   │   └── scenarios.py            #   场景 S1–S4（由易到难）
│   ├── agents/
│   │   ├── base.py                 #   控制器统一接口 select_action(obs)
│   │   ├── discretizer.py          #   观测 → 离散状态（供表格法使用）
│   │   ├── q_learning.py           #   表格型 Q-learning
│   │   └── dqn_agent.py            #   DQN（第二阶段，基于 stable-baselines3）
│   ├── controllers/
│   │   ├── fixed_time.py           #   基线一：固定时长控制
│   │   └── queue_based.py          #   基线二：基于队列控制（LQF）
│   ├── metrics.py                  #   等待时间、队列长度、吞吐量等指标
│   ├── train.py                    #   训练入口
│   └── evaluate.py                 #   评估入口（控制器 × 场景 × 种子）
├── experiments/
│   └── configs/                    # 场景与超参数配置                          🚧
├── results/                        # 实验输出 CSV                              🚧
└── report/                         # 图表与最终报告                            🚧
```

**数据流**：`nets/`（路网与需求）→ `src/env/`（封装为 Gymnasium 环境）→ `src/agents/`＋`src/controllers/`（各控制器）→ `src/train.py` / `src/evaluate.py` → `results/`（CSV）→ `report/`（图表与报告）。

---

# 环境配置 / Environment Setup

## 1. 前置要求

| 组件 | 要求 | 说明 |
| --- | --- | --- |
| Python | ≥ 3.9（本项目在 3.11 下开发） | `sumo-rl` 要求 |
| SUMO | ≥ 1.14（建议 1.20 及以上） | `sumo-rl` 依赖 `sumolib`/`traci` ≥ 1.14 |
| 操作系统 | Windows / Linux / macOS | 命令按平台分别给出 |

> 注意：`pip install` 只装 SUMO 的 **Python 接口**，`sumo` / `sumo-gui` **可执行程序必须单独安装**，并通过 `SUMO_HOME` 与 `PATH` 暴露给 Python。

## 2. 安装 SUMO

**Windows**：从 <https://sumo.dlr.de/docs/Downloads.php> 下载`.msi`安装包，默认安装到
`C:\Program Files (x86)\Eclipse\Sumo`。

安装时勾选前两项添加到`PATH`中


**验证**：

```bash
sumo --version
sumo-gui --version
```


## 3. 安装 Python 依赖

创建并激活虚拟环境：

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\activate


> 若 PowerShell 报"禁止运行脚本"，先执行：
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`
```
安装依赖：
务必在虚拟环境下执行：(.venv) PS E:\desktop_directory\desktop\AY10>
```bash
pip install -r .\requirements.txt
```

`sumo-rl` 会自动带入 `gymnasium`、`pettingzoo`、`traci`、`sumolib`、`pillow`。

进入 DQN 阶段后再安装：

```bash
pip install "stable-baselines3[extra]>=2.0"
```

## 4. 验证安装

```bash
python scripts/check.env.py #检查库环境
python scripts/demo_fixed_time.py #检查仿真sumo是否正常运行
```


若第二条报 `SUMO_HOME` 相关错误，请回到第 3 步检查环境变量，并确认已重开终端。

## 5. 常用补充

**训练加速（可选）**：使用 libsumo 替代 TraCI，约 8 倍加速，但此时**不能**使用 `sumo-gui`，也不能并行跑多个仿真。

```powershell
$env:LIBSUMO_AS_TRACI=1     # Windows，当前会话有效
```

```bash
export LIBSUMO_AS_TRACI=1   # Linux / macOS
```

建议：训练时开启，调参与演示时关闭。

**可视化检查路网/信号**：

```bash
sumo-gui -c nets/single-intersection/single-intersection.sumocfg
```


---

# Title
基于强化学习的交通信号控制
Traffic Signal Control with Reinforcement Learning

# Description
创建一个简化的单路口仿真，并训练一个表格型 Q-learning 智能体来选择红绿灯相位。将其与固定时长控制和基于排队的控制策略在不同交通流模式下进行比较，测量平均等待时间和队列长度。
Create a simplified simulation of a single intersection and train a tabular Q-learning agent to select traffic light phases. Compare it with fixed-time and queue-based control policies under different traffic patterns, measuring average waiting time and queue length.

# Environment
构建一个简化版的单个交叉口仿真，车辆从不同方向到达并在队列中等待。你可以先从两个交通信号相位和简单的车辆到达模式开始。
Build a simplified simulation of a single intersection,where vehicles arrive from different directions and wait in queues. You can begin with two traffic light phases and simple vehicle arrival patterns.

# RL Agent
训练一个表格型Q学习智能体来选择交通信号相位。状态可以包括队列长度和当前信号相位，而奖励可以惩罚等待时间或过长队列。表格型Q学习将动作价值存储在表中，不需要神经网络，因此它提供了一个相对容易上手的起点。如果你们小组对深度强化学习感兴趣，可以选择性地将项目扩展到深度Q网络（DQN），并比较这两种方法。
Train a tabular Q-learning agent to select traffic light phases. The state could include queue lengths and the current light phase, while the reward could penalise waiting time or long queues. Tabular Q-learning stores action values in a table and does not require a neural network, so itprovides a relatively accessible starting point. If your group is interested in deep reinforcement learning, you could optionally extend the project to a Deep Q-Network (DQN) and compare the two approaches.  

# Baseline
将你的智能体与固定时间控制器以及基于队列的控制器进行比较。固定时间控制器按固定时间表切换信号灯，基于队列的控制器则优先放行等待车辆更多的方向。
Compare your agent with a fixed-time controller that switches lights on a regular schedule and a queue-based controller that prioritises the direction with more waiting vehicles

# Evaluate
在不同交通条件下进行测试，例如轻度交通、重度交通或各方向车流不均衡。平均等待时间和队列长度是有用的指标。
Test under different traffic conditions, such as light traffic, heavy traffic, or uneven flows across directions. Average waiting time and queue length are useful metrics.