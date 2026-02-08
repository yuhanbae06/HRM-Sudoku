# grpo_finetune.py
import torch
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List
import random

from safetensors.torch import load_file, save_file
from adam_atan2_pytorch import AdamAtan2

from sudoku import load_online_puzzle
from train import TrainingBatch, train_step
from model import HRMACTInner, HRMACTModelConfig


# =========================================================
# Rollout buffer
# =========================================================

@dataclass
class RolloutSample:
    hidden_states: Tuple[torch.Tensor, torch.Tensor]  # (low, high) [1,82,H]
    board_inputs: torch.Tensor                         # [81]
    old_logp: torch.Tensor                             # scalar
    entropy: torch.Tensor                              # scalar
    prediction: torch.Tensor                           # [81]
    reward: torch.Tensor                               # scalar


# =========================================================
# Sample-level reward
# =========================================================

def compute_sample_reward(prediction, board_inputs, board_targets):
    mask = (board_inputs == 0)
    correct = ((prediction == board_targets) & mask).float().sum()
    denom = mask.float().sum().clamp(min=1)
    return correct / denom


# =========================================================
# Dataset-backed batched rollout
# =========================================================

class RolloutDatasetSampler:
    """
    Lightweight dataset sampler for GRPO rollout.
    Uses the same dataset format as TrainingBatch.
    """
    def __init__(self, shard="train[:1%]"):
        self.dataset = load_online_puzzle(shard)
        self.size = len(self.dataset)
        self.cursor = len(self.dataset) // 2  # start from middle to reduce correlation

    def sample_batch(self, batch_size):
        puzzles, solutions = [], []
        for _ in range(batch_size):
            idx = self.cursor % self.size
            sample = self.dataset[idx]
            puzzles.append(sample["puzzle"].flatten())
            solutions.append(sample["solution"].flatten())
            self.cursor += 1
        return puzzles, solutions


def collect_batched_rollout_output_policy(
    model,
    device,
    dataset_sampler: RolloutDatasetSampler,
    batch_size: int,
):
    """
    Batched rollout using dataset puzzles.
    ACT = environment, output logits = policy.
    """
    config = model.config
    buffer: List[RolloutSample] = []

    puzzles, solutions = dataset_sampler.sample_batch(batch_size)

    # Convert lists of numpy arrays to stacked tensors
    board_inputs = torch.tensor(np.array(puzzles), dtype=torch.long, device=device)     # [B,81]
    board_targets = torch.tensor(np.array(solutions), dtype=torch.long, device=device)  # [B,81]

    low_h = model.initial_low_level.unsqueeze(0).expand(
        batch_size, config.seq_len + 1, -1
    ).to(device, dtype=config.dtype)
    high_h = model.initial_high_level.unsqueeze(0).expand(
        batch_size, config.seq_len + 1, -1
    ).to(device, dtype=config.dtype)
    hidden_states = (low_h, high_h)

    for _ in range(config.act.halt_max_steps):

        if board_inputs.size(0) == 0:
            break

        hs_snapshot = (
            hidden_states[0].detach().clone(),
            hidden_states[1].detach().clone(),
        )

        new_hidden, output_logits, q_h, q_c = model(hidden_states, board_inputs)

        dist = torch.distributions.Categorical(logits=output_logits)
        actions = dist.sample()  # [B_alive,81]

        mask = (board_inputs == 0).float()  # Ensure float for multiplication
        logp = (dist.log_prob(actions) * mask).sum(dim=1)  # [B]
        entropy = (dist.entropy() * mask).sum(dim=1)  # [B]

        rewards = torch.stack([
            compute_sample_reward(actions[i], board_inputs[i], board_targets[i])
            for i in range(actions.size(0))
        ])

        for i in range(actions.size(0)):
            buffer.append(
                RolloutSample(
                    hidden_states=(
                        hs_snapshot[0][i:i+1],
                        hs_snapshot[1][i:i+1],
                    ),
                    board_inputs=board_inputs[i].detach().clone(),
                    old_logp=logp[i].detach(),
                    entropy=entropy[i].detach(),
                    prediction=actions[i].detach().clone(),
                    reward=rewards[i].detach(),
                )
            )

        hidden_states = (
            new_hidden[0].detach(),
            new_hidden[1].detach(),
        )

        # ACT termination (environment)
        alive_mask = (q_h <= q_c)  # [B]
        if not alive_mask.all():
            board_inputs = board_inputs[alive_mask]  # [B_new, 81]
            board_targets = board_targets[alive_mask]  # [B_new, 81]
            hidden_states = (
                new_hidden[0][alive_mask],  # Use new_hidden, not old hidden_states
                new_hidden[1][alive_mask],
            )

    return buffer


# =========================================================
# GRPO update
# =========================================================

def grpo_advantages_from_samples(rewards, group_size):
    assert rewards.numel() % group_size == 0
    r = rewards.view(-1, group_size)
    adv = r - r.mean(dim=1, keepdim=True)
    return adv.view(-1)


def grpo_update_from_buffer(
    model,
    optimizer,
    buffer,
    group_size,
    clip_eps=0.2,
    entropy_coef=0.01,
    max_grad_norm=1.0,
):
    device = next(model.parameters()).device

    rewards = torch.stack([s.reward for s in buffer]).to(device)
    advantages = grpo_advantages_from_samples(rewards, group_size).detach()

    logp_old, logp_new, entropy = [], [], []

    for sample in buffer:
        hs = (
            sample.hidden_states[0].to(device),
            sample.hidden_states[1].to(device),
        )
        board = sample.board_inputs.unsqueeze(0).to(device)

        _, output_logits, _, _ = model(hs, board)
        dist = torch.distributions.Categorical(logits=output_logits)

        mask = (board == 0).float()  # Ensure float for multiplication
        pred = sample.prediction.unsqueeze(0).to(device)

        logp_new.append((dist.log_prob(pred) * mask).sum())
        entropy.append((dist.entropy() * mask).sum())
        logp_old.append(sample.old_logp.to(device))

    logp_old = torch.stack(logp_old)
    logp_new = torch.stack(logp_new)
    entropy = torch.stack(entropy)

    ratio = torch.exp(logp_new - logp_old)
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)

    policy_loss = -torch.mean(torch.min(ratio * advantages, clipped * advantages))
    loss = policy_loss - entropy_coef * entropy.mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()

    return {
        "loss": loss.item(),
        "avg_reward": rewards.mean().item(),
        "kl": (logp_old - logp_new).mean().abs().item(),
    }


# =========================================================
# Unified GRPO + Q-ACT training
# =========================================================

def train_grpo_act(
    checkpoint_path: str,
    iterations: int = 1000,
    grpo_rollout_batch_size: int = 32,
    grpo_group_size: int = 8,
    grpo_lr: float = 5e-5,
    dataset_shard: str = "train[:1%]",
    save_every: int = 250,
):
    # ----------------------------
    # Device / model / config
    # ----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = HRMACTModelConfig(
        seq_len=81,
        vocab_size=10,
        high_level_cycles=2,
        low_level_cycles=2,
        transformers=HRMACTModelConfig.TransformerConfig(
            num_layers=4, hidden_size=256, num_heads=4, expansion=4
        ),
        act=HRMACTModelConfig.ACTConfig(
            halt_max_steps=16,
            halt_exploration_probability=0.1
        )
    )

    model = HRMACTInner(config).to(device, dtype=config.dtype)
    model.load_state_dict(load_file(checkpoint_path, device=str(device)))
    model.train()

    # ----------------------------
    # Optimizer (shared - both GRPO and Q-ACT use same optimizer)
    # Note: Two separate optimizers with different LRs would conflict on same params
    # Using grpo_lr; if different rates needed, implement param groups
    # ----------------------------
    optimizer = AdamAtan2(
        model.parameters(), lr=grpo_lr, betas=(0.9, 0.95)
    )

    # ----------------------------
    # Dataset-backed samplers
    # ----------------------------
    rollout_sampler = RolloutDatasetSampler(dataset_shard)

    qact_batch = TrainingBatch(
        model=model,
        batch_size=128,
        device=device,
        shard=dataset_shard
    )

    # ----------------------------
    # Training loop
    # ----------------------------
    for it in range(1, iterations + 1):

        # (1) GRPO step
        buffer = collect_batched_rollout_output_policy(
            model=model,
            device=device,
            dataset_sampler=rollout_sampler,
            batch_size=grpo_rollout_batch_size,
        )

        grpo_stats = grpo_update_from_buffer(
            model=model,
            optimizer=optimizer,
            buffer=buffer,
            group_size=grpo_group_size,
        )

        # (2) Q-ACT step
        qact_acc = train_step(
            model=model,
            optimizer=optimizer,
            batch=qact_batch
        )

        if it % 10 == 0:
            print(
                f"[Iter {it}] "
                f"GRPO R={grpo_stats['avg_reward']:.3f} "
                f"KL={grpo_stats['kl']:.4f} | "
                f"Q-ACT Acc={qact_acc:.3f}"
            )

        if it % save_every == 0:
            path = f"checkpoint-grpo-act-{it}.safetensors"
            save_file(model.state_dict(), path)
            print(f"Saved {path}")
