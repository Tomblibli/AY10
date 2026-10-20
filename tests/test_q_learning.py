"""表格型 Q-learning 单元测试（不依赖 SUMO）。"""

import numpy as np

from src.agents.q_learning import TabularQLearning


def _identity(obs):
    """把观测四舍五入后当作状态，便于构造确定的测试用例。"""
    return tuple(np.round(np.asarray(obs, dtype=float), 2))


def _agent(**kwargs):
    defaults = dict(n_actions=2, discretize=_identity, alpha=0.5, gamma=0.0,
                    epsilon=0.0, epsilon_min=0.0, epsilon_decay=1.0, seed=0)
    defaults.update(kwargs)
    return TabularQLearning(**defaults)


def test_select_action_within_range():
    agent = _agent(n_actions=4)
    for _ in range(50):
        assert 0 <= agent.select_action(np.zeros(3)) < 4


def test_update_moves_toward_reward():
    agent = _agent()
    obs = np.array([1.0])
    agent.update(obs, 0, 1.0, obs, terminated=True)
    assert agent.Q[(1.0,)][0] == 0.5
    agent.update(obs, 0, 1.0, obs, terminated=True)
    assert agent.Q[(1.0,)][0] == 0.75


def test_update_uses_bootstrap_when_not_terminated():
    agent = _agent(gamma=0.5)
    obs = np.array([1.0])
    nxt = np.array([2.0])
    agent.Q[(2.0,)] = np.array([4.0, 0.0])
    # target = 1.0 + 0.5 * max(4, 0) = 3.0 ; Q 从 0 起算 → 0 + 0.5*3 = 1.5
    agent.update(obs, 0, 1.0, nxt, terminated=False)
    assert agent.Q[(1.0,)][0] == 1.5


def test_greedy_action_picks_max_q():
    agent = _agent()
    obs = np.array([0.0])
    agent.Q[tuple(obs)] = np.array([0.1, 9.9])
    assert agent.select_action(obs) == 1


def test_epsilon_decays_to_floor():
    agent = _agent(epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.5)
    for _ in range(10):
        agent.on_episode_end()
    assert agent.epsilon == 0.1


def test_save_and_load_roundtrip(tmp_path):
    agent = _agent()
    obs = np.array([3.0])
    agent.update(obs, 1, 2.0, obs, terminated=True)

    path = tmp_path / "q.pkl"
    agent.save(path)

    restored = _agent()
    restored.load(path)
    assert restored.Q[(3.0,)][1] == agent.Q[(3.0,)][1]
    assert restored.n_states == 1
