#!/usr/bin/env python3
"""Train PPO on CartPole-v1 (CPU) and save matplotlib learning curves."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim

from ppo import PolicyNet, ValueNet, compute_gae, ppo_update


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PPO on CartPole-v1 (PyTorch CPU)")
    p.add_argument("--total-timesteps", type=int, default=300_000)
    p.add_argument("--num-steps", type=int, default=2048, help="Rollout length per update")
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--minibatch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-interval", type=int, default=5, help="Save plots every N updates")
    p.add_argument("--plot-dir", type=str, default="plots")
    p.add_argument(
        "--checkpoint-path",
        type=str,
        default="checkpoints/policy.pt",
        help="Where to save policy weights after training (for eval_cartpole.py)",
    )
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_plots(
    plot_dir: Path,
    episode_returns: list[float],
    updates: list[int],
    policy_loss: list[float],
    value_loss: list[float],
    entropy: list[float],
    approx_kl: list[float],
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    ax = axes[0, 0]
    ax.plot(episode_returns, alpha=0.35, label="return")
    if len(episode_returns) >= 10:
        w = min(50, len(episode_returns) // 5)
        kernel = np.ones(w) / w
        smoothed = np.convolve(episode_returns, kernel, mode="valid")
        ax.plot(range(w - 1, w - 1 + len(smoothed)), smoothed, label=f"MA({w})")
    ax.set_xlabel("episode")
    ax.set_ylabel("episode return")
    ax.legend(loc="lower right")
    ax.set_title("Episode returns")

    ax = axes[0, 1]
    if updates:
        ax.plot(updates, policy_loss, label="policy loss")
        ax.set_xlabel("update")
        ax.set_ylabel("loss")
        ax.set_title("Policy loss")
        ax.legend()

    ax = axes[1, 0]
    if updates:
        ax.plot(updates, value_loss, color="C1", label="value loss")
        ax.set_xlabel("update")
        ax.set_ylabel("loss")
        ax.set_title("Value loss")
        ax.legend()

    ax = axes[1, 1]
    if updates:
        ax.plot(updates, entropy, label="entropy", color="C2")
        ax2 = ax.twinx()
        ax2.plot(updates, approx_kl, label="approx KL", color="C3", alpha=0.8)
        ax.set_xlabel("update")
        ax.set_ylabel("entropy")
        ax2.set_ylabel("approx KL")
        ax.set_title("Entropy & approx KL")
        lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
        labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
        ax.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    path = plot_dir / "training_curves.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    device = torch.device("cpu")
    set_seed(args.seed)

    plot_dir = Path(args.plot_dir)
    env = gym.make("CartPole-v1")
    obs_space = env.observation_space
    assert hasattr(obs_space, "shape")
    obs_dim = int(obs_space.shape[0])
    n_actions = int(env.action_space.n)

    hidden_dim = 64
    policy = PolicyNet(obs_dim, n_actions, hidden=hidden_dim).to(device)
    value_net = ValueNet(obs_dim, hidden=hidden_dim).to(device)
    optimizer = optim.Adam(
        list(policy.parameters()) + list(value_net.parameters()),
        lr=args.lr,
        eps=1e-5,
    )

    episode_returns: list[float] = []
    updates_log: list[int] = []
    policy_loss_log: list[float] = []
    value_loss_log: list[float] = []
    entropy_log: list[float] = []
    kl_log: list[float] = []

    global_step = 0
    update_idx = 0
    obs, _ = env.reset(seed=args.seed)
    ep_return = 0.0
    start_time = time.time()

    while global_step < args.total_timesteps:
        max_roll = min(args.num_steps, args.total_timesteps - global_step)

        obs_buf = np.zeros((max_roll, obs_dim), dtype=np.float32)
        actions_buf = np.zeros(max_roll, dtype=np.int64)
        rewards_buf = np.zeros(max_roll, dtype=np.float32)
        dones_buf = np.zeros(max_roll, dtype=np.float32)
        log_probs_buf = np.zeros(max_roll, dtype=np.float32)
        values_buf = np.zeros(max_roll, dtype=np.float32)
        next_obs_buf = np.zeros((max_roll, obs_dim), dtype=np.float32)

        t = 0
        while t < max_roll and global_step < args.total_timesteps:
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, log_prob, _ = policy.act(obs_t)
                val = value_net(obs_t).squeeze(0)

            a = int(action.item())
            next_obs, reward, terminated, truncated, _ = env.step(a)
            done = terminated or truncated

            obs_buf[t] = obs
            actions_buf[t] = a
            rewards_buf[t] = float(reward)
            dones_buf[t] = float(done)
            log_probs_buf[t] = float(log_prob.item())
            values_buf[t] = float(val.item())
            next_obs_buf[t] = next_obs

            ep_return += float(reward)
            global_step += 1
            obs = next_obs

            if done:
                episode_returns.append(ep_return)
                ep_return = 0.0
                obs, _ = env.reset()

            t += 1

        num_steps = t
        obs_buf = obs_buf[:num_steps]
        actions_buf = actions_buf[:num_steps]
        rewards_buf = rewards_buf[:num_steps]
        dones_buf = dones_buf[:num_steps]
        log_probs_buf = log_probs_buf[:num_steps]
        values_buf = values_buf[:num_steps]
        next_obs_buf = next_obs_buf[:num_steps]

        obs_th = torch.as_tensor(obs_buf, dtype=torch.float32, device=device)
        actions_th = torch.as_tensor(actions_buf, dtype=torch.int64, device=device)
        rewards_th = torch.as_tensor(rewards_buf, dtype=torch.float32, device=device)
        dones_th = torch.as_tensor(dones_buf, dtype=torch.float32, device=device)
        log_probs_old = torch.as_tensor(log_probs_buf, dtype=torch.float32, device=device)
        values_th = torch.as_tensor(values_buf, dtype=torch.float32, device=device)

        with torch.no_grad():
            next_obs_th = torch.as_tensor(next_obs_buf, dtype=torch.float32, device=device)
            next_values_raw = value_net(next_obs_th)
            next_values = next_values_raw * (1.0 - dones_th)

        advantages, returns = compute_gae(
            rewards_th,
            values_th,
            next_values,
            dones_th,
            args.gamma,
            args.gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        metrics = ppo_update(
            policy,
            value_net,
            optimizer,
            obs_th,
            actions_th,
            log_probs_old,
            advantages,
            returns,
            clip_coef=args.clip_coef,
            vf_coef=args.vf_coef,
            ent_coef=args.ent_coef,
            num_epochs=args.num_epochs,
            minibatch_size=args.minibatch_size,
        )

        update_idx += 1
        updates_log.append(update_idx)
        policy_loss_log.append(metrics["policy_loss"])
        value_loss_log.append(metrics["value_loss"])
        entropy_log.append(metrics["entropy"])
        kl_log.append(metrics["approx_kl"])

        if update_idx % args.log_interval == 0 or global_step >= args.total_timesteps:
            save_plots(
                plot_dir,
                episode_returns,
                updates_log,
                policy_loss_log,
                value_loss_log,
                entropy_log,
                kl_log,
            )
            elapsed = time.time() - start_time
            mean_ret = float(np.mean(episode_returns[-20:])) if len(episode_returns) >= 20 else float(
                np.mean(episode_returns)
            ) if episode_returns else 0.0
            print(
                f"update {update_idx}  step {global_step}  "
                f"pi_loss {metrics['policy_loss']:.4f}  v_loss {metrics['value_loss']:.4f}  "
                f"H {metrics['entropy']:.3f}  kl~ {metrics['approx_kl']:.4f}  "
                f"episodes {len(episode_returns)}  mean_ret_20 {mean_ret:.1f}  ({elapsed:.0f}s)"
            )

    env.close()
    save_plots(
        plot_dir,
        episode_returns,
        updates_log,
        policy_loss_log,
        value_loss_log,
        entropy_log,
        kl_log,
    )
    ckpt_path = Path(args.checkpoint_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "obs_dim": obs_dim,
            "n_actions": n_actions,
            "hidden": hidden_dim,
        },
        ckpt_path,
    )
    print(f"Done. Plots: {plot_dir.resolve() / 'training_curves.png'}")
    print(f"Saved policy: {ckpt_path.resolve()}")


if __name__ == "__main__":
    main()
