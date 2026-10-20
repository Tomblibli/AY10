#!/usr/bin/env python
"""用 sumo-gui 演示「固定时长」信号控制（纯动画，不涉及 RL）。

固定时长策略直接来自路网自带的 ``static tlLogic``：相位按固定 duration
循环切换，不感知交通状态。本脚本只做两件事：

1. 启动 ``sumo-gui`` 打开可视化窗口；
2. 逐步推进仿真，并在终端打印**相位切换时刻**与在途车辆数，
   方便和动画画面一一对应。

用法（在项目根目录执行）::

    python scripts/demo_fixed_time.py                      # 600s，GUI，约 30 秒动画
    python scripts/demo_fixed_time.py --duration 300 --delay 100
    python scripts/demo_fixed_time.py --no-gui              # 无窗口自检（不开 GUI）
    python scripts/demo_fixed_time.py --net-dir nets/single-intersection

参数说明
--------
* ``--duration``：仿真时长（仿真秒）。
* ``--delay``   ：sumo-gui 每步停留毫秒数。``delay=50`` 即 1 仿真秒用 50ms 展示。
* ``--no-gui``  ：用无界面的 ``sumo`` 运行，用于快速验证脚本本身。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def drop_libsumo_flag() -> None:
    """libsumo 无法启动 GUI，若存在该环境变量则在本进程内移除。"""
    if os.environ.pop("LIBSUMO_AS_TRACI", None) is not None:
        print("[demo] 检测到 LIBSUMO_AS_TRACI=1，已在本进程内移除（libsumo 不支持 GUI）。")


def find_binary(name: str) -> str | None:
    """在 PATH 或 ``SUMO_HOME/bin`` 中查找 sumo / sumo-gui。"""
    found = shutil.which(name)
    if found:
        return found

    home = os.environ.get("SUMO_HOME")
    if home:
        suffix = ".exe" if os.name == "nt" else ""
        candidate = Path(home) / "bin" / f"{name}{suffix}"
        if candidate.is_file():
            return str(candidate)
    return None


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 sumo-gui 演示固定时长信号控制")
    parser.add_argument("--duration", type=int, default=600, help="仿真时长（秒，默认 600）")
    parser.add_argument("--delay", type=int, default=50, help="每步停留毫秒数（默认 50）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--report-every", type=int, default=100, help="每多少步打印一次状态")
    parser.add_argument("--net-dir", default=None, help="自定义路网目录（默认自动查找）")
    parser.add_argument("--no-gui", action="store_true", help="不打开 GUI，用 sumo 运行")
    return parser.parse_args(argv)


def describe_program(traci, tls_id: str) -> None:
    """打印该信号灯的固定配时相位表。"""
    logics = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logics:
        return
    logic = logics[0]
    print(f"\n[demo] 信号灯 {tls_id!r} 的固定配时（programID={logic.programID}）：")
    for index, phase in enumerate(logic.phases):
        print(f"  相位 {index}: duration={phase.duration:>5.1f}s   state={phase.state}")
    print()


def main(argv=None) -> int:
    args = parse_args(argv)
    drop_libsumo_flag()

    try:
        import traci
    except Exception as exc:  # noqa: BLE001
        print(f"[demo] 无法导入 traci：{exc!r}")
        print("       请先执行 pip install -r requirements.txt")
        return 1

    from src.env.sumo_env import resolve_net_files

    try:
        net_file, route_file = resolve_net_files(args.net_dir)
    except FileNotFoundError as exc:
        print(f"[demo] {exc}")
        return 1

    print(f"[demo] 路网: {net_file}")
    print(f"[demo] 需求: {route_file}")

    binary_name = "sumo" if args.no_gui else "sumo-gui"
    binary = find_binary(binary_name)
    if binary is None:
        print(f"[demo] 未找到 {binary_name}。")
        print("       若本机只安装了 sumo（无 GUI），请加 --no-gui 运行。")
        return 1

    cmd = [
        binary,
        "-n", str(net_file),
        "-r", str(route_file),
        "--seed", str(args.seed),
        "--no-step-log", "true",
        "--no-warnings", "true",
    ]
    if not args.no_gui:
        # --start 立即开始（否则 GUI 会停在暂停状态）；--delay 控制动画速度
        cmd += ["--start", "--delay", str(args.delay)]

    traci.start(cmd)
    try:
        tls_ids = list(traci.trafficlight.getIDList())
        if not tls_ids:
            print("[demo] 该路网没有信号灯，无法演示。")
            return 1
        tls_id = tls_ids[0]
        describe_program(traci, tls_id)

        mode = "sumo-gui" if not args.no_gui else "sumo（无界面）"
        print(f"[demo] 开始仿真：{mode}，时长 {args.duration}s，步长 1s")

        prev_state = None
        wait_total, wait_samples = 0.0, 0
        arrived_total = 0
        step_index = 0

        while traci.simulation.getTime() < args.duration:
            traci.simulationStep()
            sim_time = traci.simulation.getTime()
            arrived_total += traci.simulation.getArrivedNumber()

            state = traci.trafficlight.getRedYellowGreenState(tls_id)
            if state != prev_state:
                print(f"  t={sim_time:6.1f}s  相位切换 -> state={state}")
                prev_state = state

            vehicles = traci.vehicle.getIDList()
            if vehicles:
                wait_total += sum(traci.vehicle.getWaitingTime(v) for v in vehicles)
                wait_samples += len(vehicles)

            if step_index % max(1, args.report_every) == 0:
                print(f"  t={sim_time:6.1f}s  在途车辆={len(vehicles):3d}")
            step_index += 1

        mean_wait = wait_total / wait_samples if wait_samples else 0.0
        print("\n[demo] 仿真结束")
        print(f"  仿真时长      : {traci.simulation.getTime():.0f}s")
        print(f"  累计到达车辆  : {arrived_total}")
        print(f"  平均等待时间  : {mean_wait:.2f}s（逐步平均）")
        return 0
    except KeyboardInterrupt:
        print("\n[demo] 已手动中断。")
        return 130
    finally:
        try:
            traci.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    raise SystemExit(main())
