"""AY10 项目源代码包。

模块划分::

    src/env/          仿真环境封装（SUMO + SUMO-RL）
    src/agents/       RL 智能体（表格型 Q-learning / DQN / 状态离散化）
    src/controllers/  基线控制器（固定时长 / 基于队列）
    src/metrics.py    指标计算
    src/train.py      训练入口
    src/evaluate.py   评估入口
"""

__version__ = "0.1.0"
