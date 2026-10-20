"""场景与配置模块单元测试（不依赖 SUMO）。"""

import json
from pathlib import Path

import pytest

from src.config import DEFAULT_CONFIG, deep_update, env_kwargs, load_config, resolve_config_path
from src.env.scenarios import (
    DIRECTIONS,
    build_route_file,
    get_scenario,
    list_scenarios,
    per_direction_demand,
    summary_table,
)


# --------------------------------------------------------------------------- #
# 场景
# --------------------------------------------------------------------------- #
def test_directions_order():
    assert DIRECTIONS == ("N", "E", "S", "W")


def test_get_scenario_is_case_insensitive():
    assert get_scenario("s1")["total_demand"] == 600


def test_get_scenario_unknown_raises():
    with pytest.raises(KeyError):
        get_scenario("S99")


def test_per_direction_demand_balanced():
    demand = per_direction_demand("S1")
    assert set(demand) == set(DIRECTIONS)
    assert demand["N"] == pytest.approx(150.0)
    assert sum(demand.values()) == pytest.approx(600.0)


def test_per_direction_demand_uneven():
    demand = per_direction_demand("S4")  # 3:1:1:1, 总 1200
    assert demand["N"] == pytest.approx(600.0)
    assert demand["E"] == pytest.approx(200.0)


def test_summary_table_mentions_all_scenarios():
    table = summary_table()
    for name in list_scenarios():
        assert name in table


def test_build_route_file_is_not_implemented_yet():
    with pytest.raises(NotImplementedError):
        build_route_file("S1", "unused.rou.xml")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def test_load_config_returns_independent_copy():
    first = load_config()
    first["episodes"] = 999
    assert load_config()["episodes"] == DEFAULT_CONFIG["episodes"]


def test_deep_update_merges_nested_dicts():
    base = {"env": {"a": 1, "b": 2}}
    deep_update(base, {"env": {"b": 20, "c": 30}})
    assert base == {"env": {"a": 1, "b": 20, "c": 30}}


def test_load_config_from_file_merges_with_defaults(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"episodes": 7, "env": {"delta_time": 1}}), encoding="utf-8")

    config = load_config(path)
    assert config["episodes"] == 7
    assert config["env"]["delta_time"] == 1
    # 未覆盖的键保留默认值
    assert config["env"]["num_seconds"] == DEFAULT_CONFIG["env"]["num_seconds"]
    assert config["agent"]["alpha"] == DEFAULT_CONFIG["agent"]["alpha"]


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "missing.json")


def test_env_kwargs_extracts_env_section():
    config = load_config()
    kwargs = env_kwargs(config, num_seconds=60)
    assert kwargs["num_seconds"] == 60
    assert kwargs["reward_fn"] == DEFAULT_CONFIG["reward_fn"]
    assert kwargs["delta_time"] == DEFAULT_CONFIG["env"]["delta_time"]


def test_resolve_config_path_finds_bundled_sample():
    path = resolve_config_path("sample.json")
    assert path.is_file()
    assert path.name == "sample.json"
