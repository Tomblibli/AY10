"""控制器统一接口。

RL 智能体（``TabularQLearning`` / ``DQNAgent``）与基线控制器
（``FixedTimeController`` / ``QueueBasedController``）都实现本接口，
从而可以在 ``train.py`` / ``evaluate.py`` 中互换。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseController(ABC):
    """所有控制器的抽象基类。

    Attributes:
        name: 控制器名称（用于日志与结果文件命名）。
    """

    name: str = "base"

    # ------------------------------------------------------------------ #
    # 必须实现
    # ------------------------------------------------------------------ #
    @abstractmethod
    def select_action(self, obs: Any) -> int:
        """根据观测选择动作（SUMO-RL 中为"下一个绿灯相位"的索引）。"""

    # ------------------------------------------------------------------ #
    # 可选钩子（有默认实现，子类按需覆盖）
    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """在每个 episode 开始时调用，用于清空控制器内部计数。"""

    def update(
        self,
        obs: Any,
        action: int,
        reward: float,
        next_obs: Any,
        terminated: bool,
    ) -> None:
        """在线学习接口；基线控制器保持空实现。"""

    def on_episode_end(self) -> None:
        """在每个 episode 结束时调用（例如衰减探索率）。"""

    # ------------------------------------------------------------------ #
    # 持久化（Q 表 / 网络权重）
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} 未实现 save()")

    def load(self, path: str | Path) -> None:
        raise NotImplementedError(f"{type(self).__name__} 未实现 load()")

    def __repr__(self) -> str:  # pragma: no cover - 仅用于调试输出
        return f"<{type(self).__name__} name={self.name!r}>"


__all__ = ["BaseController"]
