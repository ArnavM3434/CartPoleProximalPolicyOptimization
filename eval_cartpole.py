#!/usr/bin/env python3
"""Load a saved PPO policy and run CartPole-v1 with a visible pygame window."""

from __future__ import annotations

import argparse
from pathlib import Path

import gymnasium as gym
import torch

from ppo import PolicyNet


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch a trained policy play CartPole")
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/policy.pt"),
        help="Path from train_ppo.py --checkpoint-path",
    )
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions from the policy; default is greedy (argmax)",
    )
    return p.parse_args()


def load_policy(path: Path, device: torch.device) -> PolicyNet:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    obs_dim = int(ckpt["obs_dim"])
    n_actions = int(ckpt["n_actions"])
    hidden = int(ckpt.get("hidden", 64))
    policy = PolicyNet(obs_dim, n_actions, hidden=hidden).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")

    if not args.checkpoint.is_file():
        raise SystemExit(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Train first: python train_ppo.py  (saves policy to checkpoints/policy.pt by default)"
        )

    policy = load_policy(args.checkpoint, device)
    env = gym.make("CartPole-v1", render_mode="human")

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = policy.act(obs_t, deterministic=not args.stochastic)
            obs, reward, terminated, truncated, _ = env.step(int(action.item()))
            ep_return += float(reward)

        print(f"episode {ep + 1}/{args.episodes}  return {ep_return:.0f}")

    env.close()


if __name__ == "__main__":
    main()
