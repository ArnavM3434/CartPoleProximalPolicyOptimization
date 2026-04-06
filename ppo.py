"""separate policy and value networks, GAE, clipped update."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer: nn.Module, std: float = 1.0, bias_const: float = 0.0) -> nn.Module:
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class PolicyNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, n_actions), std=0.01),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)

    def act(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = logits.argmax(dim=-1)
        else:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy


class ValueNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized advantage estimation. All inputs shape (T,).

    `next_values` must be V(s') already multiplied by (1 - done) so timeouts
    and terminations do not bootstrap from an autoreset observation.
    """
    t_steps = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros((), device=rewards.device, dtype=rewards.dtype)
    for t in range(t_steps - 1, -1, -1):
        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_values[t] - values[t]
        gae = delta + gamma * lam * non_terminal * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(
    policy: PolicyNet,
    value_net: ValueNet,
    optimizer: torch.optim.Optimizer,
    obs: torch.Tensor,
    actions: torch.Tensor,
    log_probs_old: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    num_epochs: int,
    minibatch_size: int,
) -> dict[str, float]:
    """One PPO update pass over stored rollout (already on device)."""
    t_steps = obs.shape[0]
    idx = torch.arange(t_steps, device=obs.device)

    total_pi_loss = 0.0
    total_v_loss = 0.0
    total_ent = 0.0
    total_kl = 0.0
    n_minibatches = 0

    for _ in range(num_epochs):
        perm = idx[torch.randperm(t_steps, device=obs.device)]
        for start in range(0, t_steps, minibatch_size):
            mb = perm[start : start + minibatch_size]
            b_obs = obs[mb]
            b_actions = actions[mb]
            b_log_old = log_probs_old[mb]
            b_adv = advantages[mb]
            b_ret = returns[mb]

            logits = policy(b_obs)
            dist = Categorical(logits=logits)
            log_prob = dist.log_prob(b_actions)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_prob - b_log_old)
            surr1 = ratio * b_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * b_adv
            pi_loss = -torch.min(surr1, surr2).mean()

            values_pred = value_net(b_obs)
            v_loss = 0.5 * ((values_pred - b_ret) ** 2).mean()

            loss = pi_loss + vf_coef * v_loss - ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_net.parameters()), 0.5)
            optimizer.step()

            with torch.no_grad():
                approx_kl = (b_log_old - log_prob).mean().item()

            total_pi_loss += pi_loss.item()
            total_v_loss += v_loss.item()
            total_ent += entropy.item()
            total_kl += approx_kl
            n_minibatches += 1

    n = max(n_minibatches, 1)
    return {
        "policy_loss": total_pi_loss / n,
        "value_loss": total_v_loss / n,
        "entropy": total_ent / n,
        "approx_kl": total_kl / n,
    }
