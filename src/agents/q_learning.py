"""表格型 Q-learning 智能体。

Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]

状态由 :class:`~src.agents.discretizer.Discretizer` 离散化得到，
Q 表用稀疏字典存储（未见过的状态按全零初始化）。

.. note::
   超参数初值取自 plan.md §4.3，尚未针对 4 相位路网调优。
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from .base import BaseController


class TabularQLearning(BaseController):
    """ε-greedy 表格型 Q-learning。

    Args:
        n_actions: 动作个数（绿灯相位个数）。
        discretize: 观测 → 离散状态（可哈希元组）的映射函数。
        alpha: 学习率。
        gamma: 折扣因子。
        epsilon: 初始探索率。
        epsilon_min: 探索率下限。
        epsilon_decay: 每个 episode 结束后的衰减系数。
        seed: 随机种子（保证可复现）。

    Example:
        >>> agent = TabularQLearning(n_actions=4, discretize=lambda o: tuple(np.round(o, 1)))
        >>> isinstance(agent.select_action(np.zeros(8)), (int, np.integer))
        True
    """

    name = "tabular_q_learning"

    def __init__(
        self,
        n_actions: int,
        discretize: Callable[[Any], Tuple],
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 0,
    ) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions 必须为正整数")
        self.n_actions = int(n_actions)
        self.discretize = discretize
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)
        self.rng = np.random.default_rng(seed)

        #: 稀疏 Q 表：state -> ndarray(n_actions)
        self.Q: Dict[Tuple, np.ndarray] = {}

    # ------------------------------------------------------------------ #
    # 内部工具
    # ------------------------------------------------------------------ #
    def _q(self, state: Tuple) -> np.ndarray:
        """返回状态对应的 Q 值向量（按需创建）。"""
        if state not in self.Q:
            self.Q[state] = np.zeros(self.n_actions, dtype=float)
        return self.Q[state]

    # ------------------------------------------------------------------ #
    # BaseController 接口
    # ------------------------------------------------------------------ #
    def select_action(self, obs: Any) -> int:
        """ε-greedy 选择动作。"""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))

        q_values = self._q(self.discretize(obs))
        best = np.flatnonzero(q_values == q_values.max())
        # 平局随机打破，避免固定偏好某个相位
        return int(self.rng.choice(best))

    def update(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
    ) -> None:
        """执行一次 Q-learning 更新。"""
        state = self.discretize(obs)
        next_state = self.discretize(next_obs)

        target = float(reward)
        if not terminated:
            target += self.gamma * float(np.max(self._q(next_state)))

        q_values = self._q(state)
        q_values[action] += self.alpha * (target - q_values[action])

    def on_episode_end(self) -> None:
        """每个 episode 结束后衰减探索率。"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # 持久化
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        """把 Q 表与超参数写入 pickle 文件。"""
        payload = {
            "Q": self.Q,
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
        }
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as fh:
            pickle.dump(payload, fh)

    def load(self, path: str | Path) -> None:
        """从 pickle 文件恢复 Q 表与超参数。"""
        with Path(path).open("rb") as fh:
            payload = pickle.load(fh)
        self.Q = payload["Q"]
        self.n_actions = payload["n_actions"]
        self.alpha = payload["alpha"]
        self.gamma = payload["gamma"]
        self.epsilon = payload["epsilon"]
        self.epsilon_min = payload["epsilon_min"]
        self.epsilon_decay = payload["epsilon_decay"]

    # ------------------------------------------------------------------ #
    # 统计信息
    # ------------------------------------------------------------------ #
    @property
    def n_states(self) -> int:
        """已访问过的状态数（用于观察状态空间覆盖情况）。"""
        return len(self.Q)


__all__ = ["TabularQLearning"]
