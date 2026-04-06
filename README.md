# CartPole PPO (PyTorch, CPU)

Proximal Policy Optimization **implemented from scratch** in PyTorch for [Gymnasium](https://gymnasium.farama.org/) **CartPole-v1**. Training runs on **CPU**; learning curves are saved with Matplotlib; a trained policy can be watched in a pygame window.

## Design

### Actor–critic with two networks

- **Policy network (`PolicyNet`)** — MLP: 4 → 64 → 64 → 2 with ReLU. Outputs **logits** for the two discrete actions. State includes 4 numbers - cart position, cart velocity, pole angle (from vertical), and pole angular speed.
- **Value network (`ValueNet`)** — Separate MLP: 4 → 64 → 64 → 1 with ReLU. Predicts **V(s)** for bootstrapping and the critic loss.

Both networks use **orthogonal weight initialization** and **zero biases** (`layer_init` in `ppo.py`) for stable early updates in RL.

### PPO objective

- **Clipped surrogate** on the probability ratio \(r = \pi_\text{new}/\pi_\text{old}\) with **clip coefficient** ε (default `0.2`).
- **Value loss**: MSE between predicted **V(s)** and **GAE targets** (returns = advantages + values).
- **Entropy bonus** weighted by `ent_coef` to encourage exploration early in training.
- **Gradient clipping** (norm 0.5) on the combined policy + value parameters.

One **Adam** optimizer (default lr `3e-4`, `eps=1e-5`) updates **both** networks.

### GAE (λ)

Advantages are **Generalized Advantage Estimation** with discount **γ** (`gamma`) and **λ** (`gae-lambda`). The implementation walks **backward** along the rollout, using **`dones`** so credit does not propagate across episode boundaries.

**Autoreset:** after a terminal or truncated step, Gymnasium may return the **first observation of the next episode**. The code sets **next-state values to zero when `done`** before GAE, so we never bootstrap from that autoreset observation.

### Rollout strategy

Each PPO **update** uses a fixed budget of environment steps **`num-steps`** (default **2048**), except the last rollout may be shorter so the run stops exactly at **`total-timesteps`**.

A rollout is **not** “one episode.” It is **up to `num-steps` consecutive `env.step` calls**. If CartPole fails early, **`env.reset()`** is called and the **same rollout buffer** keeps filling until the step budget is met. So one rollout usually contains **many partial or full trajectories** in one flat time series; episode structure is encoded only in **`dones`**.

After each rollout: compute **GAE**, **normalize advantages** (zero mean, unit variance), then run **`ppo_update`** for **`num-epochs`** full passes over that data in shuffled **minibatches** (`minibatch-size`). Then the buffer is discarded and new on-policy data is collected.

## Setup

```bash
cd /path/to/CartPole
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`gymnasium[classic-control]` pulls in **pygame**, needed for **`eval_cartpole.py`** window rendering.

## Run training

```bash
python train_ppo.py
```

**Outputs**

- **`plots/training_curves.png`** — episode returns (with a simple moving average when enough episodes exist), policy loss, value loss, entropy, and approximate KL vs update index. Refreshed every **`log-interval`** updates and at the end.
- **`checkpoints/policy.pt`** — policy weights and metadata (`obs_dim`, `n_actions`, `hidden`) after training completes.

Useful flags:

```bash
python train_ppo.py --total-timesteps 150000 --seed 1 --log-interval 3
python train_ppo.py --checkpoint-path checkpoints/my_policy.pt --plot-dir runs/plots
```

Full CLI: `python train_ppo.py --help`.

## Run visual evaluation

After training (checkpoint exists):

```bash
python eval_cartpole.py
python eval_cartpole.py --episodes 10 --checkpoint checkpoints/policy.pt
```

Default behavior uses **greedy** actions (argmax). Add **`--stochastic`** to sample from the policy. Requires a **display** (local desktop or working GUI forwarding).

## Important hyperparameters

| Parameter | Default | Role |
|-----------|---------|------|
| **`total-timesteps`** | 300000 | Total `env.step` calls for the whole run. |
| **`num-steps`** | 2048 | Environment steps collected **per PPO update** (rollout length). |
| **`num-epochs`** | 10 | How many **full passes** over **that same rollout** before new data (not “10 policies”). |
| **`minibatch-size`** | 64 | Minibatch size inside each epoch. |
| **`gamma`** | 0.99 | Discount factor in GAE / bootstrap. |
| **`gae-lambda`** | 0.95 | GAE bias–variance tradeoff (higher → more Monte-Carlo-like advantages). |
| **`clip-coef`** | 0.2 | PPO clip ε on the probability ratio. |
| **`lr`** | 3e-4 | Adam learning rate for policy + value. |
| **`vf-coef`** | 0.5 | Scale of the value loss relative to the policy term. |
| **`ent-coef`** | 0.01 | Entropy bonus strength (exploration). |
| **`seed`** | 0 | NumPy / PyTorch / first env reset seed. |
| **`log-interval`** | 5 | Save `training_curves.png` every N updates. |

Rough count: number of outer PPO updates ≈ **`ceil(total-timesteps / num-steps)`**. Optimizer steps per update ≈ **`num-epochs × ceil(rollout_length / minibatch-size)`**.

## What you typically see

### Episode return

CartPole-v1 gives **+1 reward per timestep** until failure or **time limit**. The maximum return in a single episode is therefore **500** when the pole stays balanced for the full **500-step** horizon (truncation). In practice, a good run’s **recent episode returns** and **moving average** climb toward that cap; the classic “solved” benchmark is often quoted as **average return ≥ 475 over 100 episodes**, but **500** is the hard ceiling for a single episode under the default time limit.

Early training often shows **short episodes** (returns on the order of tens); as the policy improves, returns **increase sharply** into the hundreds.

### Entropy

For two actions, the **maximum** categorical entropy is **log 2 ≈ 0.693** nats (uniform random left/right). As the policy becomes **more confident**, entropy **decreases** toward **0** (nearly deterministic). The **`ent_coef`** term slows premature collapse; if entropy drops too fast while return is still poor, slightly **increasing `ent_coef`** can help exploration (at the risk of slower convergence).

### Policy loss and approximate KL

The logged **policy loss** is the **negative** clipped surrogate (what gets added into the total loss), so **more negative** often correlates with a **larger** policy improvement signal on that batch—interpret it relative to its own scale and alongside **approx KL**. **Approximate KL** (mean log π_old − log π_new on the batch) tends to stay **small** when clipping is doing its job; **spikes** can mean the update is aggressive relative to the behavior policy.

### Value loss

**Value loss** can be **noisy** and **larger in magnitude** than the policy term because it is a squared error on returns; it often **does not monotonically decrease** step-to-step. What matters for CartPole is whether **episode returns** improve while training remains stable.

## Project layout

| File | Purpose |
|------|---------|
| `ppo.py` | `PolicyNet`, `ValueNet`, `compute_gae`, `ppo_update`. |
| `train_ppo.py` | Rollout loop, logging, plots, checkpoint save. |
| `eval_cartpole.py` | Load checkpoint, `render_mode="human"`. |
| `requirements.txt` | PyTorch, Gymnasium (classic control), Matplotlib, NumPy. |

`plots/` and `checkpoints/` are listed in `.gitignore` by default.

