"""训练入口。

用法（在项目根目录执行）::

    python src/train.py --controller random --episodes 1 --dry-run
    python src/train.py --controller qlearning --episodes 50
    python src/train.py --controller qlearning --episodes 200 --reward queue --seed 7

说明
----
* ``--dry-run`` 只构建环境并打印空间信息，不做训练，适合先验证环境是否可用；
* 当前日志只记录**回合累计奖励**与**步数**；
  平均等待时间 / 队列长度需接入 :mod:`src.metrics` 的 CSV 读取（待实现）；
* 表格型 Q-learning 需要离散化，观测布局假设见 :mod:`src.agents.discretizer`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 允许以 `python src/train.py` 直接运行时找到 `src` 包
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import Discretizer, TabularQLearning  # noqa: E402
from src.agents.base import BaseController  # noqa: E402
from src.controllers import FixedTimeController  # noqa: E402
from src.env import (  # noqa: E402
    action_count,
    build_env,
    describe_env,
    reset_env,
    step_env,
)
from src.env.scenarios import summary_table  # noqa: E402


class RandomController(BaseController):
    """随机策略：用于验证环境可用性与作为性能下界。"""

    name = "random"

    def __init__(self, n_actions: int, seed: int = 0) -> None:
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)

    def select_action(self, obs) -> int:  # noqa: D102
        return int(self.rng.integers(self.n_actions))


def build_controller(name: str, env, args: argparse.Namespace) -> BaseController:
    """按名称构造控制器。"""
    n_actions = action_count(env)

    if name == "random":
        return RandomController(n_actions, seed=args.seed)

    if name == "fixed_time":
        return FixedTimeController(n_actions, switch_every=args.switch_every)

    if name == "qlearning":
        obs_dim = int(env.observation_space.shape[0])
        discretizer = Discretizer.from_spaces(n_phases=n_actions, obs_dim=obs_dim)
        print(f"[train] 离散化: {discretizer}")
        return TabularQLearning(
            n_actions=n_actions,
            discretize=discretizer,
            alpha=args.alpha,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_decay=args.epsilon_decay,
            seed=args.seed,
        )

    raise ValueError(f"未知控制器 {name!r}；可选: random / fixed_time / qlearning")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练交通信号控制智能体")
    parser.add_argument(
        "--controller", default="random", choices=["random", "fixed_time", "qlearning"],
        help="控制器类型（默认 random，用于快速验证）",
    )
    parser.add_argument("--episodes", type=int, default=1, help="训练回合数")
    parser.add_argument("--max-steps", type=int, default=10_000, help="单回合最大步数（保护）")
    parser.add_argument("--reward", default="diff-waiting-time",
                        help="SUMO-RL 奖励函数名，如 diff-waiting-time / queue")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--gui", action="store_true", help="开启 sumo-gui 可视化")
    parser.add_argument("--num-seconds", type=int, default=3600, help="每回合仿真时长（秒）")
    parser.add_argument("--delta-time", type=int, default=5, help="决策间隔（秒）")
    parser.add_argument("--switch-every", type=int, default=6, help="固定时长控制的切换步数")
    # Q-learning 超参数
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--dry-run", action="store_true", help="只建环境并打印信息，不训练")
    parser.add_argument("--save", default=None, help="训练结束后保存模型路径（qlearning）")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    print("[train] 场景一览：")
    print(summary_table())

    env = build_env(
        use_gui=args.gui,
        num_seconds=args.num_seconds,
        delta_time=args.delta_time,
        reward_fn=args.reward,
    )
    try:
        print("\n[train] 环境信息：")
        print(describe_env(env))

        if args.dry_run:
            print("\n[train] --dry-run：环境构建成功，跳过训练。")
            return 0

        controller = build_controller(args.controller, env, args)
        print(f"\n[train] 控制器: {controller}")

        for episode in range(1, args.episodes + 1):
            obs, _ = reset_env(env)
            controller.reset()

            total_reward, steps = 0.0, 0
            for _ in range(args.max_steps):
                action = controller.select_action(obs)
                next_obs, reward, terminated, truncated, _ = step_env(env, action)
                controller.update(obs, action, reward, next_obs, terminated)

                obs = next_obs
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    break

            controller.on_episode_end()
            extra = ""
            if isinstance(controller, TabularQLearning):
                extra = f" | epsilon={controller.epsilon:.3f} | states={controller.n_states}"
            print(
                f"[episode {episode:4d}] return={total_reward:10.2f} | "
                f"steps={steps:4d}{extra}"
            )

        if args.save and hasattr(controller, "save"):
            controller.save(args.save)
            print(f"\n[train] 模型已保存到 {args.save}")

        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
