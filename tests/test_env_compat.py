"""环境封装层的兼容性测试（用假环境对象，不需要 SUMO）。"""

from types import SimpleNamespace

import numpy as np

from src.env.sumo_env import action_count, filter_supported_kwargs, reset_env, step_env


class OldStyleEnv:
    """旧版 gym 风格：reset 只返回 obs，step 返回 4 元组。"""

    action_space = SimpleNamespace(n=4)
    observation_space = SimpleNamespace(shape=(9,))

    def reset(self):
        return np.zeros(9)

    def step(self, action):
        return np.zeros(9), 1.5, False, {}


class NewStyleEnv:
    """Gymnasium 风格：reset 返回 (obs, info)，step 返回 5 元组。"""

    action_space = SimpleNamespace(n=2)
    observation_space = SimpleNamespace(shape=(5,))

    def reset(self):
        return np.zeros(5), {"reset": True}

    def step(self, action):
        return np.zeros(5), 2.0, False, True, {"step": action}


def test_reset_env_old_style_returns_tuple():
    obs, info = reset_env(OldStyleEnv())
    assert obs.shape == (9,)
    assert info == {}


def test_reset_env_new_style_passthrough():
    obs, info = reset_env(NewStyleEnv())
    assert obs.shape == (5,)
    assert info == {"reset": True}


def test_step_env_old_style_normalises_to_five_tuple():
    obs, reward, terminated, truncated, info = step_env(OldStyleEnv(), 0)
    assert obs.shape == (9,)
    assert reward == 1.5
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_step_env_new_style_passthrough():
    obs, reward, terminated, truncated, info = step_env(NewStyleEnv(), 1)
    assert reward == 2.0
    assert terminated is False
    assert truncated is True
    assert info == {"step": 1}


def test_action_count_from_discrete_space():
    assert action_count(OldStyleEnv()) == 4


def test_action_count_from_multidiscrete_like_space():
    env = SimpleNamespace(action_space=SimpleNamespace(nvec=[3]))
    assert action_count(env) == 3


def test_filter_supported_kwargs_drops_unknown_keys():
    class Dummy:
        def __init__(self, a, b=1):
            self.a, self.b = a, b

    filtered = filter_supported_kwargs(Dummy, {"a": 1, "b": 2, "unknown": 3})
    assert filtered == {"a": 1, "b": 2}


def test_filter_supported_kwargs_keeps_all_when_var_keyword():
    class Flexible:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    filtered = filter_supported_kwargs(Flexible, {"anything": 1})
    assert filtered == {"anything": 1}
