"""指标模块单元测试（不依赖 SUMO，使用真实列名构造数据）。"""

import math
from pathlib import Path

import pandas as pd
import pytest

from src.metrics import (
    METRIC_FIELDS,
    aggregate_seeds,
    episode_csv_path,
    find_episode_csv,
    load_and_summarize,
    summarize_dataframe,
    summarize_episode,
)


# --------------------------------------------------------------------------- #
# 数组版汇总
# --------------------------------------------------------------------------- #
def test_summarize_episode_basic():
    result = summarize_episode([1, 2, 3], [0, 2, 4])
    assert result["avg_wait"] == 2.0
    assert result["avg_queue"] == 2.0
    assert result["max_queue"] == 4.0
    assert result["n_samples"] == 3


def test_summarize_episode_empty_returns_nan():
    result = summarize_episode([], [])
    for field in ("avg_wait", "avg_queue", "max_queue", "p95_wait"):
        assert math.isnan(result[field])


# --------------------------------------------------------------------------- #
# CSV 版汇总（列名取自 SUMO-RL 实测输出）
# --------------------------------------------------------------------------- #
def _fake_episode_df():
    return pd.DataFrame({
        "step": [0, 5, 10],
        "system_mean_waiting_time": [0.0, 2.0, 4.0],
        "system_total_stopped": [0.0, 3.0, 6.0],
        "t_accumulated_waiting_time": [0.0, 5.0, 15.0],
        "t_average_speed": [10.0, 8.0, 6.0],
    })


def test_summarize_dataframe_maps_real_columns():
    result = summarize_dataframe(_fake_episode_df())
    assert result["avg_wait"] == 2.0
    assert result["avg_queue"] == 3.0
    assert result["max_queue"] == 6.0
    assert result["total_wait"] == 15.0
    assert result["mean_speed"] == 8.0
    assert result["duration"] == 10.0
    assert result["n_samples"] == 3


def test_summarize_dataframe_handles_missing_columns():
    result = summarize_dataframe(pd.DataFrame({"step": [0, 5]}))
    assert math.isnan(result["avg_wait"])
    assert result["n_samples"] == 2


def test_metric_fields_are_all_present():
    result = summarize_dataframe(_fake_episode_df())
    assert set(METRIC_FIELDS) <= set(result)


# --------------------------------------------------------------------------- #
# 文件定位与读取
# --------------------------------------------------------------------------- #
def test_episode_csv_path_follows_sumo_rl_convention():
    path = episode_csv_path("results/run", episode=2)
    assert path == Path("results/run_conn0_ep2.csv")


def test_find_episode_csv(tmp_path):
    target = tmp_path / "run_conn0_ep3.csv"
    target.write_text("step\n0\n", encoding="utf-8")
    assert find_episode_csv(tmp_path / "run", episode=3) == target


def test_find_episode_csv_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_episode_csv(tmp_path / "nope", episode=0)


def test_load_and_summarize_reads_file(tmp_path):
    path = tmp_path / "ep.csv"
    _fake_episode_df().to_csv(path, index=False)
    assert load_and_summarize(path)["avg_queue"] == 3.0


# --------------------------------------------------------------------------- #
# 跨种子汇总
# --------------------------------------------------------------------------- #
def test_aggregate_seeds_mean_std_count():
    runs = [{"avg_wait": 1.0, "avg_queue": 2.0},
            {"avg_wait": 3.0, "avg_queue": 4.0}]
    stats = aggregate_seeds(runs, fields=("avg_wait", "avg_queue"))
    assert stats["avg_wait"]["mean"] == 2.0
    assert stats["avg_wait"]["std"] == pytest.approx(1.414, abs=1e-3)
    assert stats["avg_wait"]["n"] == 2


def test_aggregate_seeds_skips_nan():
    runs = [{"avg_wait": 1.0}, {"avg_wait": float("nan")}]
    stats = aggregate_seeds(runs, fields=("avg_wait",))
    assert stats["avg_wait"]["mean"] == 1.0
    assert stats["avg_wait"]["n"] == 1
