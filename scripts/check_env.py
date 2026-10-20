#!/usr/bin/env python
"""环境自检脚本：确认 SUMO、Python 依赖与 SUMO-RL 是否配置成功。

用法（在项目根目录执行）::

    python scripts/check_env.py            # 完整检查（含一次约 60 秒仿真试跑）
    python scripts/check_env.py --no-sim   # 只做静态检查，不启动仿真

退出码：``0`` 表示关键检查全部通过；``1`` 表示至少一项失败。

检查项
------
1. Python 版本（要求 ``>= 3.9``）
2. 环境变量 ``SUMO_HOME`` 是否设置且目录存在
3. ``sumo`` 可执行文件是否可找到并能输出版本
4. Python 包是否可导入：numpy / pandas / gymnasium / traci / sumolib
5. ``sumo_rl`` 是否可导入、路网文件是否可定位
6. 真正启动一次 SUMO 仿真并执行若干步（最关键的一步）
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
RESULTS: list[tuple[str, str, str]] = []

REQUIRED_PACKAGES = ("numpy", "pandas", "gymnasium", "traci", "sumolib")
OPTIONAL_PACKAGES = ("matplotlib",)


def record(name: str, status: str, detail: str = "") -> None:
    """记录并立即打印一条检查结果。"""
    RESULTS.append((name, status, detail))
    suffix = f"  ->  {detail}" if detail else ""
    print(f"[{status:4s}] {name}{suffix}", flush=True)


# --------------------------------------------------------------------------- #
# 各项检查
# --------------------------------------------------------------------------- #
def check_python() -> bool:
    version = sys.version_info
    ok = version >= (3, 9)
    record(
        "Python 版本",
        PASS if ok else FAIL,
        f"{version.major}.{version.minor}.{version.micro}"
        f"（要求 >= 3.9，当前平台 {platform.system()}）",
    )
    return ok


def check_sumo_home() -> Path | None:
    home = os.environ.get("SUMO_HOME")
    if not home:
        record(
            "SUMO_HOME",
            FAIL,
            "未设置。sumo-rl 在 import 阶段就会因缺少该变量而报错。",
        )
        return None

    path = Path(home)
    if not path.is_dir():
        record("SUMO_HOME", FAIL, f"目录不存在: {path}")
        return None

    record("SUMO_HOME", PASS, str(path))
    if not (path / "tools").is_dir():
        record("SUMO_HOME/tools", WARN, f"未找到 {path / 'tools'}")
    return path


def check_sumo_binary(home: Path | None) -> str | None:
    exe_name = "sumo.exe" if os.name == "nt" else "sumo"
    exe = shutil.which("sumo")
    if exe is None and home is not None:
        candidate = home / "bin" / exe_name
        if candidate.is_file():
            exe = str(candidate)

    if exe is None:
        record(
            "sumo 可执行文件",
            FAIL,
            "未找到。请把 SUMO_HOME\\bin 加入 PATH，或检查 SUMO 是否安装完成。",
        )
        return None

    try:
        proc = subprocess.run(
            [exe, "--version"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        lines = (proc.stdout or proc.stderr or "").strip().splitlines()
        detail = lines[0] if lines else f"exit={proc.returncode}"
        record("sumo 可执行文件", PASS, f"{exe} | {detail}")
        return exe
    except Exception as exc:  # noqa: BLE001 - 自检脚本需要兜底
        record("sumo 可执行文件", FAIL, f"调用 {exe} 失败: {exc!r}")
        return None


def check_packages() -> bool:
    all_ok = True
    for name in REQUIRED_PACKAGES:
        try:
            module = importlib.import_module(name)
            version = getattr(module, "__version__", "unknown")
            record(f"import {name}", PASS, f"version={version}")
        except Exception as exc:  # noqa: BLE001
            record(f"import {name}", FAIL, f"{type(exc).__name__}: {exc}")
            all_ok = False

    for name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(name)
            record(f"import {name}", PASS)
        except Exception as exc:  # noqa: BLE001
            record(f"import {name}", WARN, f"未安装（非必需）: {type(exc).__name__}")
    return all_ok


def check_sumo_rl() -> bool:
    try:
        import sumo_rl
    except Exception as exc:  # noqa: BLE001
        record(
            "import sumo_rl",
            FAIL,
            f"{type(exc).__name__}: {exc}。"
            "常见原因：SUMO_HOME 未设置，或未执行 pip install sumo-rl。",
        )
        return False

    record("import sumo_rl", PASS, f"version={getattr(sumo_rl, '__version__', 'unknown')}")

    try:
        from src.env.sumo_env import resolve_net_files

        net_file, route_file = resolve_net_files()
        record("路网文件定位", PASS, str(net_file.parent))
        record("  net.xml", PASS, net_file.name)
        record("  rou.xml", PASS, route_file.name)
        return True
    except Exception as exc:  # noqa: BLE001
        record("路网文件定位", FAIL, f"{type(exc).__name__}: {exc}")
        return False


def check_simulation() -> bool:
    """真正启动一次 SUMO 仿真（最关键的一步）。"""
    try:
        from src.env.sumo_env import (
            action_count,
            build_env,
            describe_env,
            reset_env,
            step_env,
        )
    except Exception as exc:  # noqa: BLE001
        record("加载 src.env.sumo_env", FAIL, f"{type(exc).__name__}: {exc}")
        return False

    env = None
    try:
        env = build_env(num_seconds=60, use_gui=False, out_csv_name=None)
        obs, _ = reset_env(env)
        n_actions = action_count(env)
        obs_dim = len(obs) if hasattr(obs, "__len__") else -1

        rewards = []
        for _ in range(3):
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, _ = step_env(env, action)
            rewards.append(round(reward, 3))
            if terminated or truncated:
                break

        record(
            "SUMO 仿真试跑",
            PASS,
            f"obs_dim={obs_dim}, actions={n_actions}, steps={len(rewards)}, rewards={rewards}",
        )
        print("\n环境摘要：")
        print("  " + describe_env(env).replace("\n", "\n  "))
        return True
    except Exception as exc:  # noqa: BLE001
        record("SUMO 仿真试跑", FAIL, f"{type(exc).__name__}: {exc}")
        return False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001
                pass


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="检查 SUMO / SUMO-RL 环境是否配置成功")
    parser.add_argument("--no-sim", action="store_true", help="跳过仿真试跑")
    args = parser.parse_args(argv)

    print("=" * 68)
    print("AY10 环境自检")
    print(f"项目根目录: {PROJECT_ROOT}")
    print("=" * 68)

    check_python()
    home = check_sumo_home()
    sumo_exe = check_sumo_binary(home)
    pkg_ok = check_packages()
    sumo_rl_ok = check_sumo_rl()

    if args.no_sim:
        record("SUMO 仿真试跑", SKIP, "已通过 --no-sim 跳过")
    elif not (home and sumo_exe and pkg_ok and sumo_rl_ok):
        record("SUMO 仿真试跑", SKIP, "前置检查未通过，跳过")
    else:
        check_simulation()

    # ---- 汇总 ----
    print("\n" + "=" * 68)
    counts = {status: 0 for status in (PASS, FAIL, WARN, SKIP)}
    for _, status, _ in RESULTS:
        counts[status] = counts.get(status, 0) + 1
    print(
        f"汇总: PASS={counts[PASS]}  FAIL={counts[FAIL]}  "
        f"WARN={counts[WARN]}  SKIP={counts[SKIP]}"
    )

    failed = [name for name, status, _ in RESULTS if status == FAIL]
    if failed:
        print("\n未通过的检查：")
        for name in failed:
            print(f"  - {name}")
        print("\n排查建议：")
        print("  1) 确认已安装 SUMO，且 SUMO_HOME 指向安装目录（设置后需重开终端）")
        print("  2) 确认已执行 pip install -r requirements.txt")
        print("  3) 确认当前目录为项目根目录（含 src/ 与 nets/）")
        print("=" * 68)
        return 1

    print("\n环境检查全部通过，可以继续运行：python src/train.py --dry-run")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

