"""观测 → 离散状态（供表格型 Q-learning 使用）。

SUMO-RL 的默认观测是**连续向量**：

    obs = [phase_one_hot(n_phases), min_green(1), density(n_lanes), queue(n_lanes)]

表格型 Q-learning 需要有限状态，因此把密度与排队长度**分箱**。

.. warning::
   上述布局依据 SUMO-RL 文档（``DefaultObservationFunction``）。
   若后续改用其它观测函数（如 ``states="state"`` 的占用网格），
   需要同步修改本模块。**首次接入真实环境时应打印观测长度核对。**
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np

#: 默认分箱边界（右开区间）：0 | 1-2 | 3-5 | 6-10 | >10
DEFAULT_QUEUE_EDGES: Tuple[int, ...] = (1, 3, 6, 10)
DEFAULT_DENSITY_EDGES: Tuple[int, ...] = (1, 3, 6, 10)


class Discretizer:
    """把连续观测映射成可哈希的离散状态元组。

    Args:
        n_phases: 绿灯相位个数（= 动作空间大小）。
        n_lanes: 受控进口车道数。
        queue_edges: 排队长度分箱边界。
        density_edges: 车辆数分箱边界。

    Example:
        >>> d = Discretizer.from_spaces(n_phases=4, obs_dim=4 + 1 + 2 * 12)
        >>> isinstance(d(np.zeros(29)), tuple)
        True
    """

    def __init__(
        self,
        n_phases: int,
        n_lanes: int,
        queue_edges: Iterable[int] = DEFAULT_QUEUE_EDGES,
        density_edges: Iterable[int] = DEFAULT_DENSITY_EDGES,
    ) -> None:
        if n_phases <= 0 or n_lanes <= 0:
            raise ValueError("n_phases 与 n_lanes 必须为正整数")
        self.n_phases = int(n_phases)
        self.n_lanes = int(n_lanes)
        self.queue_edges = tuple(sorted(int(e) for e in queue_edges))
        self.density_edges = tuple(sorted(int(e) for e in density_edges))

    # ------------------------------------------------------------------ #
    # 构造辅助
    # ------------------------------------------------------------------ #
    @property
    def expected_dim(self) -> int:
        """按默认观测布局推算的观测维度。"""
        return self.n_phases + 1 + 2 * self.n_lanes

    @classmethod
    def from_spaces(cls, n_phases: int, obs_dim: int, **kwargs) -> "Discretizer":
        """由 ``动作个数`` 与 ``观测维度`` 反推车道数。

        ``obs_dim = n_phases + 1 + 2 * n_lanes``  ⇒  ``n_lanes = (obs_dim - n_phases - 1) / 2``

        Raises:
            ValueError: 维度无法整除，说明观测布局与假设不一致。
        """
        rest = int(obs_dim) - int(n_phases) - 1
        if rest <= 0 or rest % 2 != 0:
            raise ValueError(
                "观测维度与假设的布局不符："
                f"obs_dim={obs_dim}, n_phases={n_phases}，"
                "无法按 [phase_one_hot, min_green, density, queue] 解析。"
                "请先打印真实观测核对布局。"
            )
        return cls(n_phases=n_phases, n_lanes=rest // 2, **kwargs)

    # ------------------------------------------------------------------ #
    # 映射
    # ------------------------------------------------------------------ #
    def __call__(self, obs: Sequence[float]) -> Tuple[int, ...]:
        """把观测转换为离散状态元组。"""
        arr = np.asarray(obs, dtype=float).ravel()
        if arr.size != self.expected_dim:
            raise ValueError(
                f"观测长度 {arr.size} 与预期 {self.expected_dim} 不符；"
                "请核对 Discretizer 的 n_phases / n_lanes 设置。"
            )

        phase = int(np.argmax(arr[: self.n_phases]))
        min_green = int(arr[self.n_phases] > 0.5)

        rest = arr[self.n_phases + 1 :]
        density = rest[: self.n_lanes]
        queue = rest[self.n_lanes :]

        return (
            phase,
            min_green,
            *self._bin(density, self.density_edges),
            *self._bin(queue, self.queue_edges),
        )

    @staticmethod
    def _bin(values: np.ndarray, edges: Tuple[int, ...]) -> Tuple[int, ...]:
        """按边界把连续值映射为档位编号（0 表示最小档）。"""
        return tuple(int(np.searchsorted(edges, v, side="right")) for v in values)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<Discretizer n_phases={self.n_phases}, n_lanes={self.n_lanes}, "
            f"obs_dim={self.expected_dim}>"
        )


__all__ = ["Discretizer", "DEFAULT_QUEUE_EDGES", "DEFAULT_DENSITY_EDGES"]
