"""评估入口（框架版）。

用法（在项目根目录执行）::

    python src/evaluate.py --dry-run                       # 只检查矩阵与依赖
    python src/evaluate.py --controllers random fixed_time  # 跑完整矩阵

输出
----
``results/eval_<scenario>.csv``，列：``scenario, controller, seed, episode_return, steps``。

.. warning::
   **场景差异尚未生效**：S1–S4 的差异依赖各场景专属的 ``.rou.xml``
   （见 :func:`src.env.scenarios.build_route_file`，待实现）。
   当前所有场景共用同一份需求文件，因此结果只能用于**验证流程**，
   不能用来比较不同交通条件。

真实指标（平均等待时间 / 队列长度）待 :mod:`src.metrics` 的
CSV 读取实现后接入；目前仅记录回合累计奖励。
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env import action_count, build_env, describe_env, reset_env, step_env  # noqa: E402
from src.env.scenarios import SCENARIOS  # noqa: E402
# 复用 train.py 的控制器工厂，保证训练与评估的构造方式完全一致
from src.train import build_controller, parse_args as parse_train_args  # noqa: E402


def run_episode(env, controller, max_steps: int = 10_000):
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
    parser = argparse.ArgumentParser(description="在多个场景/控制器/种子上评估")
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                        help="要评估的场景（默认全部 S1–S4）")
    parser.add_argument("--controllers", nargs="+",
                        default=["random", "fixed_time"],
                        help="要评估的控制器")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0],
                        help="随机种子列表")
    parser.add_argument("--reward", default="diff-waiting-time")
    parser.add_argument("--num-seconds", type=int, default=3600)
    parser.add_argument("--delta-time", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--dry-run", action="store_true", help="只打印评估矩阵")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print("[evaluate] 评估矩阵：")
    print(f"  场景      : {args.scenarios}")
    print(f"  控制器    : {args.controllers}")
    print(f"  种子      : {args.seeds}")
    print(f"  奖励函数  : {args.reward}")
    print("\n[警告] 场景专属需求文件尚未实现，S1–S4 目前共用同一份 .rou.xml，"
          "结果仅用于验证流程。\n")

    if args.dry_run:
        return 0

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    train_defaults = parse_train_args([])  # 取控制器默认超参数

    for scenario in args.scenarios:
        rows = []
        for controller_name in args.controllers:
            for seed in args.seeds:
                train_defaults.seed = seed
                env = build_env(
                    num_seconds=args.num_seconds,
                    delta_time=args.delta_time,
                    reward_fn=args.reward,
                )
                try:
                    controller = build_controller(controller_name, env, train_defaults)
                    return_acc, steps = run_episode(env, controller, args.max_steps)
                finally:
                    env.close()

                rows.append({
                    "scenario": scenario,
                    "controller": controller_name,
                    "seed": seed,
                    "episode_return": f"{return_acc:.4f}",
                    "steps": steps,
                })
                print(f"[{scenario}] {controller_name:<12} seed={seed} "
                      f"return={return_acc:10.2f} steps={steps}")

        csv_path = out_dir / f"eval_{scenario}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[evaluate] 已写出 {csv_path}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
