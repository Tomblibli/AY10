"""固定时长控制器（基线一）。

两种可行做法
------------
1. **推荐**：直接使用 SUMO 网络里的 ``static tlLogic``（固定 ``duration`` 的相位序列），
   不需要任何外部控制——用 ``sumo -c xxx.sumocfg`` 直接跑即可。
   此时本文件仅用于在统一实验框架中占位（返回固定的相位轮转）。
2. 本文件实现的**简化版**：每 ``switch_every`` 次决策轮转到下一个绿灯相位，
   用于在同一套 Python 实验循环中与 RL 智能体并列比较。

.. note:.
   简化版按"决策步数"而非"仿真时间"计时，故其周期 = ``switch_every × delta_time`` 秒。
   如需严格的固定配时（如周期 60 s、绿信比 50:50），应改用做法 1。
"""

from __future__ import annotations

from typing import Any

from ..agents.base import BaseController


class FixedTimeController(BaseController):
    """按固定步数轮转相位的固定时长控制器。

    Args:
        n_actions: 绿灯相位个数。
        switch_every: 每多少次决策切换一次相位。
        start_action: 初始相位索引。
    """

    name = "fixed_time"

    def __init__(self, n_actions: int, switch_every: int = 6, start_action: int = 0) -> None:
        if n_actions <= 0:
            raise ValueError("n_actions 必须为正整数")
        if switch_every <= 0:
            raise ValueError("switch_every 必须为正整数")
        self.n_actions = int(n_actions)
        self.switch_every = int(switch_every)
        self.start_action = int(start_action) % int(n_actions)
        self._action = self.start_action
        self._steps = 0

    def reset(self) -> None:
        self._action = self.start_action
        self._steps = 0

    def select_action(self, obs: Any) -> int:
        # 到一个周期就轮转到下一个相位，否则保持
        if self._steps > 0 and self._steps % self.switch_every == 0:
            self._action = (self._action + 1) % self.n_actions
        self._steps += 1
        return self._action

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<FixedTimeController n_actions={self.n_actions}, "
            f"switch_every={self.switch_every}>"
        )


__all__ = ["FixedTimeController"]
