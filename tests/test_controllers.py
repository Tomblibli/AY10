"""基线控制器单元测试（不依赖 SUMO）。"""

import numpy as np
import pytest

from src.controllers.fixed_time import FixedTimeController
from src.controllers.queue_based import QueueBasedController


# --------------------------------------------------------------------------- #
# 固定时长
# --------------------------------------------------------------------------- #
def test_fixed_time_cycles_phases():
    controller = FixedTimeController(n_actions=3, switch_every=2)
    actions = [controller.select_action(None) for _ in range(6)]
    assert actions == [0, 0, 1, 1, 2, 2]


def test_fixed_time_wraps_around():
    controller = FixedTimeController(n_actions=2, switch_every=1)
    actions = [controller.select_action(None) for _ in range(4)]
    assert actions == [0, 1, 0, 1]


def test_fixed_time_reset_restores_state():
    controller = FixedTimeController(n_actions=3, switch_every=1)
    for _ in range(4):
        controller.select_action(None)
    controller.reset()
    assert controller.select_action(None) == 0


def test_fixed_time_rejects_bad_args():
    with pytest.raises(ValueError):
        FixedTimeController(n_actions=0)
    with pytest.raises(ValueError):
        FixedTimeController(n_actions=2, switch_every=0)


# --------------------------------------------------------------------------- #
# 基于队列（LQF）
# --------------------------------------------------------------------------- #
def _obs(min_green: int, queue, n_phases: int = 2):
    """构造 [one_hot, min_green, density, queue] 观测。"""
    queue = np.asarray(queue, dtype=float)
    one_hot = np.zeros(n_phases)
    one_hot[0] = 1.0
    return np.concatenate([one_hot, [min_green], np.zeros_like(queue), queue])


def test_queue_based_holds_when_min_green_not_met():
    controller = QueueBasedController(n_phases=2, n_lanes=4, phase_lanes=[[0, 1], [2, 3]])
    # 对向排队远大于当前相位，但最小绿灯未满足 → 必须保持当前相位(0)
    assert controller.select_action(_obs(0, [0, 0, 9, 9])) == 0


def test_queue_based_picks_longest_queue():
    controller = QueueBasedController(n_phases=2, n_lanes=4, phase_lanes=[[0, 1], [2, 3]])
    assert controller.select_action(_obs(1, [5, 5, 0, 1])) == 0
    assert controller.select_action(_obs(1, [0, 0, 3, 4])) == 1


def test_queue_based_validates_phase_lanes():
    with pytest.raises(ValueError):
        QueueBasedController(n_phases=2, n_lanes=4, phase_lanes=[[0, 1]])  # 组数不符
    with pytest.raises(ValueError):
        QueueBasedController(n_phases=2, n_lanes=4, phase_lanes=[[0, 1], [2, 9]])  # 越界


def test_queue_based_rejects_short_observation():
    controller = QueueBasedController(n_phases=2, n_lanes=4, phase_lanes=[[0, 1], [2, 3]])
    with pytest.raises(ValueError):
        controller.select_action(np.zeros(3))
