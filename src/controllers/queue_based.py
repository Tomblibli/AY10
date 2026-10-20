"""基于队列的控制器（基线二）：Longest-Queue-First (LQF)。

规则
----
1. 若最小绿灯尚未满足（``min_green`` 标志为 0），必须**保持**当前相位；
2. 否则计算每个绿灯相位对应车道的**排队总和**，选择排队最多的相位。

待完成
------
``phase_lanes`` 需要"相位 → 车道下标"的映射，可从 SUMO-RL 的
``TrafficSignal.green_phases`` 与受控连接（``getControlledLinks``）推导。
当前要求调用方显式传入，避免在框架阶段猜测路网结构::

    QueueBasedController(n_phases=4, n_lanes=12,
                         phase_lanes=[[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]])
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ..agents.base import BaseController


class QueueBasedController(BaseController):
    """按相位排队总和选择相位的 LQF 控制器。

    Args:
        n_phases: 绿灯相位个数。
        n_lanes: 受控进口车道数（决定观测末尾排队向量的长度）。
        phase_lanes: 每个相位对应的车道下标列表；下标针对**排队向量**。
        hold_on_min_green: 最小绿灯未满足时是否强制保持当前相位。
    """

    name = "queue_based"

    def __init__(
        self,
        n_phases: int,
        n_lanes: int,
        phase_lanes: Sequence[Sequence[int]],
        hold_on_min_green: bool = True,
    ) -> None:
        if n_phases <= 0 or n_lanes <= 0:
            raise ValueError("n_phases 与 n_lanes 必须为正整数")
        if len(phase_lanes) != n_phases:
            raise ValueError(
                f"phase_lanes 需为 {n_phases} 组（每相位一组），实际 {len(phase_lanes)} 组"
            )

        self.n_phases = int(n_phases)
        self.n_lanes = int(n_lanes)
        self.phase_lanes = [tuple(int(i) for i in idx) for idx in phase_lanes]
        self.hold_on_min_green = bool(hold_on_min_green)

        for idx in self.phase_lanes:
            if any(i < 0 or i >= self.n_lanes for i in idx):
                raise ValueError(f"phase_lanes 中存在越界下标（应在 0..{self.n_lanes - 1}）")

    def select_action(self, obs: Any) -> int:
        arr = np.asarray(obs, dtype=float).ravel()
        if arr.size < self.n_phases + 1 + self.n_lanes:
            raise ValueError(
                f"观测长度 {arr.size} 不足，至少需要 {self.n_phases + 1 + self.n_lanes}"
            )

        min_green_satisfied = bool(arr[self.n_phases] > 0.5)
        queues = arr[-self.n_lanes :]  # 观测布局：[phase_one_hot, min_green, density, queue]

        if self.hold_on_min_green and not min_green_satisfied:
            return int(np.argmax(arr[: self.n_phases]))

        pressure = [float(queues[list(idx)].sum()) if idx else 0.0 for idx in self.phase_lanes]
        return int(np.argmax(pressure))

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<QueueBasedController n_phases={self.n_phases}, "
            f"n_lanes={self.n_lanes}>"
        )


__all__ = ["QueueBasedController"]
