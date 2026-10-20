"""集成测试：需要本机安装 SUMO（未设置 SUMO_HOME 时自动跳过）。

运行方式::

    pytest -m integration        # 只跑集成测试
    pytest -m "not integration"  # 只跑单元测试（CI 默认）
"""

import os

import pytest

pytestmark = pytest.mark.integration

SUMOHOME_AVAILABLE = bool(os.environ.get("SUMO_HOME"))
requires_sumo = pytest.mark.skipif(not SUMOHOME_AVAILABLE, reason="需要 SUMO（SUMO_HOME 未设置）")


@requires_sumo
def test_resolve_net_files_falls_back_to_bundled(tmp_path):
    """本地 nets/ 为空时应回退到 sumo-rl 内置路网。"""
    from src.env.sumo_env import resolve_net_files

    net_file, route_file = resolve_net_files(tmp_path)
    assert net_file.is_file()
    assert route_file.is_file()


@requires_sumo
def test_environment_can_step():
    """真正启动一次短仿真，确认环境可用且动作空间有效。"""
    from src.env.sumo_env import action_count, build_env, reset_env, step_env

    env = build_env(num_seconds=30, use_gui=False, out_csv_name=None)
    try:
        obs, _ = reset_env(env)
        assert len(obs) > 0

        n_actions = action_count(env)
        assert n_actions >= 1

        for _ in range(2):
            obs, reward, terminated, truncated, _ = step_env(env, int(env.action_space.sample()))
            if terminated or truncated:
                break
    finally:
        env.close()
