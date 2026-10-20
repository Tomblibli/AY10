"""Discretizer 单元测试（不依赖 SUMO）。"""

import numpy as np
import pytest

from src.agents.discretizer import Discretizer


def test_from_spaces_infers_n_lanes():
    # obs = phase_one_hot(4) + min_green(1) + density(12) + queue(12)
    d = Discretizer.from_spaces(n_phases=4, obs_dim=4 + 1 + 24)
    assert d.n_phases == 4
    assert d.n_lanes == 12
    assert d.expected_dim == 29


def test_from_spaces_rejects_inconsistent_dim():
    with pytest.raises(ValueError):
        Discretizer.from_spaces(n_phases=4, obs_dim=30)  # (30-4-1) 为奇数
    with pytest.raises(ValueError):
        Discretizer.from_spaces(n_phases=4, obs_dim=5)  # 没有车道维度


def test_call_returns_expected_tuple():
    d = Discretizer(n_phases=2, n_lanes=2, queue_edges=(1, 3), density_edges=(1, 3))
    # [phase0=1, phase1=0] [min_green=1] [density 0,0] [queue 5,0]
    obs = np.array([1.0, 0.0, 1.0, 0.0, 0.0, 5.0, 0.0])
    assert d(obs) == (0, 1, 0, 0, 2, 0)


def test_binning_boundaries():
    d = Discretizer(n_phases=1, n_lanes=1, queue_edges=(1, 3), density_edges=(1,))
    # queue 恰好等于边界值时进入下一档（side="right"）
    assert d(np.array([1.0, 0.0, 0.0, 0.0]))[-1] == 0   # queue=0
    assert d(np.array([1.0, 0.0, 0.0, 1.0]))[-1] == 1   # queue=1
    assert d(np.array([1.0, 0.0, 0.0, 3.0]))[-1] == 2   # queue=3
    assert d(np.array([1.0, 0.0, 0.0, 99.0]))[-1] == 2  # 超出上界仍为最高档


def test_wrong_observation_length_raises():
    d = Discretizer(n_phases=2, n_lanes=3)
    with pytest.raises(ValueError):
        d(np.zeros(5))


def test_state_is_hashable():
    d = Discretizer(n_phases=2, n_lanes=2)
    state = d(np.zeros(d.expected_dim))
    assert isinstance(state, tuple)
    assert {state: 1}[state] == 1


def test_default_edges_follow_normalized_observation():
    """SUMO-RL 默认把观测缩放到 [0,1]，阈值必须落在该区间内。"""
    d = Discretizer(n_phases=2, n_lanes=1)
    assert d.normalized is True
    assert max(d.queue_edges) <= 1.0
    assert max(d.density_edges) <= 1.0


def test_count_based_edges_when_not_normalized():
    d = Discretizer(n_phases=2, n_lanes=1, normalized=False)
    assert d.normalized is False
    assert max(d.queue_edges) > 1.0


def test_normalized_values_are_separated_into_distinct_bins():
    """归一化观测必须能区分不同拥堵程度（回归测试）。"""
    d = Discretizer(n_phases=2, n_lanes=1)
    low = d(np.array([1.0, 0.0, 1.0, 0.05, 0.05]))
    high = d(np.array([1.0, 0.0, 1.0, 0.95, 0.95]))
    assert low != high

