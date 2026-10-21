"""SUMO 环境构建入口。

把 SUMO-RL 的 ``SumoEnvironment`` 包装成项目内统一的构建函数，
供 ``train.py`` / ``evaluate.py`` / ``scripts/check_env.py`` 复用。

设计要点
--------
1. **延迟导入**：``sumo_rl`` 只在真正构建环境时导入，
   因此未安装 SUMO 时仍可 ``import`` 本模块、执行 ``--help``。
2. **路网路径解析**：优先使用本地 ``nets/single-intersection/`` 下的路网；
   若本地尚未放入自定义路网，则回退到 ``sumo_rl`` 包内置的单路口路网。
   全部返回**绝对路径**，避免工作目录不同导致找不到文件。
3. **参数过滤**：SUMO-RL 各版本构造参数略有差异，
   :func:`filter_supported_kwargs` 会丢弃当前版本不支持的键。
4. **返回值兼容**：Gymnasium 与旧版 gym 的 ``reset`` / ``step`` 返回签名不同，
   :func:`reset_env` / :func:`step_env` 统一成 5 元组形式。
"""

from __future__ import annotations

import inspect
import os
from pathlib import Path
from typing import Any, Dict, Tuple

# src/env/sumo_env.py -> parents[0]=env, [1]=src, [2]=项目根目录
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
LOCAL_NET_DIR: Path = PROJECT_ROOT / "nets" / "single-intersection"

NET_FILE_NAME = "single-intersection.net.xml"
ROUTE_FILE_NAME = "single-intersection.rou.xml"

#: 环境变量覆盖路网目录（便于在不同机器上切换场景）
NET_DIR_ENV_VAR = "AY10_NET_DIR"

#: 默认环境参数，可被 :func:`build_env` 的关键字参数覆盖
DEFAULT_ENV_KWARGS: Dict[str, Any] = {
    "out_csv_name": None,           # None = 不写 CSV；训练/评估时再指定
    "single_agent": True,           # True -> 普通 Gymnasium 环境
    "use_gui": False,               # 训练关 GUI；演示时置 True
    "num_seconds": 3600,            # 每个 episode 的仿真时长（秒）
    "delta_time": 5,                # 决策间隔（秒）
    "yellow_time": 3,               # 黄灯时长（秒）
    "min_green": 5,                 # 最小绿灯（秒）
    "max_green": 50,                # 最大绿灯（秒）
    "reward_fn": "diff-waiting-time",
}


# --------------------------------------------------------------------------- #
# 路网路径解析
# --------------------------------------------------------------------------- #
def _bundled_net_dir() -> Path:
    """返回 sumo-rl 包内置的单路口路网目录。"""
    import sumo_rl  # 延迟导入

    return Path(sumo_rl.__file__).resolve().parent / "nets" / "single-intersection"


def resolve_net_files(net_dir: str | Path | None = None) -> Tuple[Path, Path]:
    """定位 ``.net.xml`` 与 ``.rou.xml``。

    优先级：显式 ``net_dir`` 参数 > 环境变量 ``AY10_NET_DIR`` >
    本地 ``nets/single-intersection/`` > ``sumo_rl`` 内置路网。

    Returns:
        ``(net_file, route_file)`` 的绝对路径。

    Raises:
        FileNotFoundError: 所有候选位置都找不到成对的路网/需求文件。
    """
    candidates: list[Path] = []
    if net_dir is not None:
        candidates.append(Path(net_dir))
    if os.environ.get(NET_DIR_ENV_VAR):
        candidates.append(Path(os.environ[NET_DIR_ENV_VAR]))
    candidates.append(LOCAL_NET_DIR)
    candidates.append(_bundled_net_dir())

    for directory in candidates:
        net = directory / NET_FILE_NAME
        route = directory / ROUTE_FILE_NAME
        if net.is_file() and route.is_file():
            return net.resolve(), route.resolve()

    searched = "\n".join(f"  - {d}" for d in candidates)
    raise FileNotFoundError(
        "未找到 SUMO 路网文件，已搜索以下目录：\n"
        f"{searched}\n"
        f"需要同时包含 {NET_FILE_NAME} 与 {ROUTE_FILE_NAME}。"
    )


# --------------------------------------------------------------------------- #
# 环境构建
# --------------------------------------------------------------------------- #
def filter_supported_kwargs(cls: type, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """丢弃 ``cls.__init__`` 不接受的键，兼容不同版本的 SUMO-RL。"""
    try:
        params = inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - 极端情况
        return dict(kwargs)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return dict(kwargs)

    supported = {name for name in params if name != "self"}
    dropped = sorted(set(kwargs) - supported)
    if dropped:
        print(f"[sumo_env] 当前 SUMO-RL 版本不支持以下参数，已忽略: {dropped}")
    return {k: v for k, v in kwargs.items() if k in supported}


def build_env(net_dir: str | Path | None = None, **overrides: Any):
    """构建单智能体的 SUMO 环境。

    Args:
        net_dir: 自定义路网目录；``None`` 时按 :func:`resolve_net_files` 的优先级查找。
        **overrides: 覆盖 :data:`DEFAULT_ENV_KWARGS` 中的任意参数
            （如 ``num_seconds=600``、``use_gui=True``、``reward_fn="queue"``）。

    Returns:
        ``sumo_rl.SumoEnvironment`` 实例（``single_agent=True`` 时即 Gymnasium 环境）。
    """
    from sumo_rl import SumoEnvironment  # 延迟导入：未装 SUMO 时也能 import 本模块

    net_file, route_file = resolve_net_files(net_dir)

    kwargs: Dict[str, Any] = dict(DEFAULT_ENV_KWARGS)
    kwargs.update(overrides)
    # 允许调用方显式传入 net_file/route_file（此时不使用解析结果）
    kwargs["net_file"] = str(Path(kwargs.get("net_file", net_file)).resolve())
    kwargs["route_file"] = str(Path(kwargs.get("route_file", route_file)).resolve())

    kwargs = filter_supported_kwargs(SumoEnvironment, kwargs)

    try:
        return SumoEnvironment(**kwargs)
    except (ValueError, KeyError) as exc:
        # 不同版本的 SUMO-RL 支持的奖励函数名可能不同；
        # 若因 reward_fn 构造失败，则去掉该参数重试，让 SUMO-RL 使用默认奖励。
        if "reward_fn" in kwargs:
            print(
                f"[sumo_env] 构造失败（{type(exc).__name__}: {exc}），"
                "改用 SUMO-RL 默认奖励函数重试。"
            )
            fallback = {k: v for k, v in kwargs.items() if k != "reward_fn"}
            return SumoEnvironment(**fallback)
        raise


# --------------------------------------------------------------------------- #
# 接口兼容层
# --------------------------------------------------------------------------- #
def reset_env(env) -> Tuple[Any, Dict[str, Any]]:
    """``reset()`` 统一为 ``(obs, info)``（兼容旧版只返回 obs 的实现）。"""
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        return out[0], out[1] or {}
    return out, {}


def step_env(env, action) -> Tuple[Any, float, bool, bool, Dict[str, Any]]:
    """``step()`` 统一为 ``(obs, reward, terminated, truncated, info)``。"""
    out = env.step(action)
    if len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return obs, float(reward), bool(terminated), bool(truncated), info or {}
    # 旧版 gym 风格：(obs, reward, done, info)
    obs, reward, done, info = out
    return obs, float(reward), bool(done), False, info or {}


def action_count(env) -> int:
    """返回离散动作个数（即路网中绿灯相位的数量）。"""
    space = env.action_space
    if hasattr(space, "n"):
        return int(space.n)
    nvec = getattr(space, "nvec", None)
    if nvec is not None:
        return int(nvec[0])
    raise TypeError(f"无法从 {space!r} 推断动作个数")


def describe_env(env) -> str:
    """生成环境信息摘要，供日志与自检使用。"""
    lines = [
        f"observation_space : {env.observation_space}",
        f"action_space     : {env.action_space}",
    ]
    try:
        lines.append(f"action_count     : {action_count(env)}  (绿灯相位个数)")
    except TypeError as exc:  # pragma: no cover
        lines.append(f"action_count     : 未知 ({exc})")

    ts_ids = getattr(env, "ts_ids", None)
    if ts_ids:
        lines.append(f"traffic_signals  : {list(ts_ids)}")
    return "\n".join(lines)


__all__ = [
    "PROJECT_ROOT",
    "LOCAL_NET_DIR",
    "DEFAULT_ENV_KWARGS",
    "resolve_net_files",
    "filter_supported_kwargs",
    "build_env",
    "reset_env",
    "step_env",
    "action_count",
    "describe_env",
]

