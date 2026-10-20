"""DQN 智能体（第二阶段，暂未实现）。

规划
----
* 观测直接使用 SUMO-RL 的**连续观测**，无需 :class:`Discretizer`；
* 用 ``stable-baselines3`` 的 ``DQN`` 训练（经验回放 + 目标网络已内置）：
  ``pip install "stable-baselines3[extra]>=2.0"``；
* 与表格法对照的维度：控制性能、收敛速度、跨场景泛化能力。

本文件目前只提供统一的构造函数骨架，便于 ``train.py`` / ``evaluate.py``
先按相同接口接线；具体训练逻辑待第二阶段实现。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .base import BaseController


class DQNAgent(BaseController):
    """DQN 控制器骨架（基于 stable-baselines3）。

    Args:
        env: 训练用环境（Gymnasium 接口）。
        learning_rate: 学习率。
        buffer_size: 经验回放容量。
        learning_starts: 开始学习前的随机步数。
        batch_size: 批大小。
        gamma: 折扣因子。
        seed: 随机种子。

    Raises:
        ImportError: 未安装 ``stable-baselines3`` 时给出安装提示。
    """

    name = "dqn"

    def __init__(
        self,
        env: Any,
        learning_rate: float = 1e-3,
        buffer_size: int = 100_000,
        learning_starts: int = 1_000,
        batch_size: int = 64,
        gamma: float = 0.95,
        seed: int = 0,
    ) -> None:
        try:
            from stable_baselines3 import DQN  # 延迟导入
        except ImportError as exc:  # pragma: no cover - 取决于环境
            raise ImportError(
                "DQN 需要 stable-baselines3，请先安装：\n"
                '  pip install "stable-baselines3[extra]>=2.0"'
            ) from exc

        self.model = DQN(
            "MlpPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            gamma=gamma,
            seed=seed,
            verbose=0,
        )

    def select_action(self, obs: Any) -> int:
        """用当前策略选择动作（训练时关闭探索由 SB3 内部处理）。"""
        action, _ = self.model.predict(obs, deterministic=False)
        return int(action)

    def update(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
    ) -> None:
        """DQN 的训练由 SB3 的 ``learn()`` 完成，此处无需逐样本更新。"""

    def learn(self, total_timesteps: int, **kwargs: Any) -> None:
        """调用 SB3 训练主循环。"""
        self.model.learn(total_timesteps=total_timesteps, **kwargs)

    def save(self, path: str | Path) -> None:
        """保存网络权重（SB3 会在文件名后附加 ``.zip``）。"""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(target))

    def load(self, path: str | Path) -> None:
        """加载网络权重。"""
        from stable_baselines3 import DQN

        self.model = DQN.load(str(path), env=self.model.get_env())


__all__ = ["DQNAgent"]
