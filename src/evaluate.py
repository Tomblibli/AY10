"""评估入口：控制器 × 场景 × 种子的批量实验。

用法（在项目根目录执行）::

    python src/evaluate.py --dry-run
    python src/evaluate.py --controllers random fixed_time --seeds 0 1 --episodes 1
    python src/evaluate.py --config sample.json --controllers qlearning --seeds 0 1 2

输出
----
``results/eval_<scenario>.csv``，每个 (控制器, 种子, 回合) 一行，含
``avg_wait / avg_queue / max_queue / p95_wait / total_wait / mean_speed / duration``。

.. warning::
   场景差异依赖各场景专属的 ``.rou.xml``（:func:`src.env.scenarios.build_route_file`
   尚未实现）。当前 S1–S4 共用同一份需求文件，结果只能验证流程，
   不能用于比较不同交通条件（见 docs/tasks.md 中 A 的阶段一任务）。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import env_kwargs, load_config, resolve_config_path  # noqa: E402
from src.env import build_env, reset_env, step_env  # noqa: E402
from src.env.scenarios import SCENARIOS  # noqa: E402
from src.factory import build_controller  # noqa: E402
from src.metrics import (  # noqa: E402
    METRIC_FIELDS,
    aggregate_seeds,
    find_episode_csv,
    load_and_summarize,
)


def run_episode(env, controller, max_steps: int):
    """跑一个 episode，返回 ``(累计奖励, 步数)``。"""
    obs, _ = reset_env(env)
    controller.reset()

    total_reward, steps = 0.0, 0
    for _ in range(max_steps):
        action = controller.select_action(obs)
        next_obs, reward, terminated, truncated, _ = step_env(env, action)
        controller.update(obs, action, reward, next_obs, terminated)

        obs = next_obs
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break
    return total_reward, steps


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量评估控制器")
    parser.add_argument("--config", default=None, help="配置文件（JSON）")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS))
    parser.add_argument("--controllers", nargs="+", default=["random", "fixed_time"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--episodes", type=int, default=1, help="每个组合重复回合数")
    parser.add_argument("--max-steps", type=int, default=100_000)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--dry-run", action="store_true", help="只打印评估矩阵")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    config = load_config(resolve_config_path(args.config)) if args.config else load_config()

    combos = len(args.scenarios) * len(args.controllers) * len(args.seeds) * args.episodes
    print("[evaluate] 评估矩阵：")
    print(f"  场景    : {args.scenarios}")
    print(f"  控制器  : {args.controllers}")
    print(f"  种子    : {args.seeds}")
    print(f"  每组合  : {args.episodes} 回合   合计 {combos} 次运行")
    print("\n[警告] 场景专属需求文件尚未实现，S1–S4 共用同一份 .rou.xml，"
          "结果仅用于验证流程。\n")
    if args.dry_run:
        return 0

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    episode_rows = []

    for scenario in args.scenarios:
        for controller_name in args.controllers:
            for seed in args.seeds:
                for episode in range(1, args.episodes + 1):
                    csv_prefix = out_dir / f"{scenario}_{controller_name}_s{seed}_e{episode}"
                    env = build_env(**env_kwargs(config, out_csv_name=str(csv_prefix)))
                    try:
                        controller = build_controller(controller_name, env, config, seed=seed)
                        total_reward, steps = run_episode(env, controller, args.max_steps)
                        if hasattr(env, "save_csv"):
                            env.save_csv(str(csv_prefix), 0)
                    finally:
                        env.close()

                    row = {
                        "scenario": scenario,
                        "controller": controller_name,
                        "seed": seed,
                        "episode": episode,
                        "episode_return": round(total_reward, 4),
                        "steps": steps,
                    }
                    try:
                        row.update(load_and_summarize(find_episode_csv(csv_prefix, 0)))
                    except Exception as exc:  # noqa: BLE001
                        print(f"[evaluate] {csv_prefix.name} 指标缺失: {type(exc).__name__}")
                        row.update({field: float("nan") for field in METRIC_FIELDS})

                    episode_rows.append(row)
                    wait = row.get("avg_wait", float("nan"))
                    queue = row.get("avg_queue", float("nan"))
                    print(f"[{scenario}] {controller_name:<11} seed={seed} ep={episode} "
                          f"return={total_reward:10.2f} wait={wait:.2f}s queue={queue:.2f}")

    for scenario in args.scenarios:
        rows = [r for r in episode_rows if r["scenario"] == scenario]
        if not rows:
            continue
        path = out_dir / f"eval_{scenario}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[evaluate] 已写出 {path}")

    print("\n[evaluate] 跨种子汇总（均值 ± 标准差）：")
    for scenario in args.scenarios:
        for controller_name in args.controllers:
            runs = [r for r in episode_rows
                    if r["scenario"] == scenario and r["controller"] == controller_name]
            if not runs:
                continue
            stats = aggregate_seeds(runs, fields=("avg_wait", "avg_queue", "max_queue"))
            print(f"  {scenario} / {controller_name:<11} "
                  f"avg_wait={stats['avg_wait']['mean']:.2f}±{stats['avg_wait']['std']:.2f}s  "
                  f"avg_queue={stats['avg_queue']['mean']:.2f}±{stats['avg_queue']['std']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

