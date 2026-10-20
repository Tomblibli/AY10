"""指标计算（框架版）。

评估指标（详见 plan.md §6.2）
-----------------------------
主指标：
    * ``avg_wait``  —— 平均等待时间（s/veh）
    * ``avg_queue`` —— 平均队列长度（veh）
辅助指标：
    * ``max_queue`` —— 最大队列长度
    * ``throughput``—— 吞吐量（veh/h）
    * ``avg_time_loss`` —— 平均延误（s/veh）
    * ``switch_count``  —— 相位切换次数

数据来源
--------
SUMO-RL 在构造环境时若指定 ``out_csv_name``，每个 episode 结束后会写出 CSV
（由 ``env.save_csv(...)`` 触发），其中包含逐步的等待时间、排队等统计量。
正式评估应从中读取；当前版本先提供纯函数的汇总工具，读取逻辑待接入。

.. note:.
   ``waiting_times`` / ``queue_lengths`` 目前由调用方提供，
   待确认 SUMO-RL CSV 的具体列名后再补充 ``load_episode_from_csv``。
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np

#: 汇总结果的标准字段顺序
METRIC_FIELDS = (
    "avg_wait",
    "avg_queue",
    "max_queue",
    "p95_wait",
    "n_samples",
)


def summarize_episode(
    waiting_times: Iterable[float],
    queue_lengths: Iterable[float],
) -> Dict[str, float]:
    """把单次 episode 的等待时间与队列长度汇总成指标字典。

    Args:
        waiting_times: 每辆车的等待时间（秒）。
        queue_lengths: 逐步的队列长度采样（辆）。

    Returns:
        含 ``avg_wait`` / ``avg_queue`` / ``max_queue`` / ``p95_wait`` / ``n_samples``。
        输入为空时对应字段返回 ``nan``（避免除零）。
    """
    waits = np.asarray(list(waiting_times), dtype=float)
    queues = np.asarray(list(queue_lengths), dtype=float)

    summary: Dict[str, float] = {
        "avg_wait": float(waits.mean()) if waits.size else float("nan"),
        "avg_queue": float(queues.mean()) if queues.size else float("nan"),
        "max_queue": float(queues.max()) if queues.size else float("nan"),
        "p95_wait": float(np.percentile(waits, 95)) if waits.size else float("nan"),
        "n_samples": float(waits.size),
    }
    return summary


def aggregate_seeds(
    runs: Iterable[Dict[str, float]],
    fields: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """跨随机种子汇总：返回每个字段的 ``mean`` / ``std``。

    Args:
        runs: 每个种子的指标字典（:func:`summarize_episode` 的输出）。
        fields: 要汇总的字段；``None`` 表示使用 ``METRIC_FIELDS``。

    Returns:
        ``{field: {"mean": ..., "std": ...}}``；某字段全为 ``nan`` 时返回 ``nan``。
    """
    runs = list(runs)
    keys = tuple(fields) if fields is not None else METRIC_FIELDS
    result: Dict[str, Dict[str, float]] = {}

    for key in keys:
        values = np.asarray([r.get(key, np.nan) for r in runs], dtype=float)
        if values.size == 0 or np.all(np.isnan(values)):
            result[key] = {"mean": float("nan"), "std": float("nan")}
        else:
            result[key] = {
                "mean": float(np.nanmean(values)),
                "std": float(np.nanstd(values)),
            }
    return result


def load_episode_from_csv(path: str) -> Dict[str, float]:
    """从 SUMO-RL 输出的 CSV 读取一个 episode 的指标。

    .. note::
       **待实现**：需先确认 CSV 的实际列名（如 ``system_mean_waiting_time``
       等），再决定如何聚合。可先用 ``pandas.read_csv`` 打印列名核对。
    """
    raise NotImplementedError(
        "load_episode_from_csv 待实现：请先用 pandas 打印 SUMO-RL 输出 CSV 的列名。"
    )


__all__ = ["METRIC_FIELDS", "summarize_episode", "aggregate_seeds", "load_episode_from_csv"]
