# train.py

import os
import torch
import torch.nn.functional as F
from model import HRMACTInner
from sudoku import generate_sudoku, Difficulty
import random
import numpy as np

from tqdm import tqdm

PREGENERATED_PUZZLES = {} # Dict: Difficulty -> List[Tuple[np.ndarray, np.ndarray]]
NUM_PREGENERATED_PER_DIFFICULTY = 1000 # Example: Generate 1,000 puzzles of each type once

def _generate_puzzle_pool():
    global PREGENERATED_PUZZLES
    # check pregenerated puzzles
    if os.path.exists('pregenerated_puzzles.npy'):
        PREGENERATED_PUZZLES = np.load('pregenerated_puzzles.npy', allow_pickle=True).item()
        print("✅ Loaded pre-generated puzzles.")
        return

    total_puzzles = NUM_PREGENERATED_PER_DIFFICULTY * len(TrainingBatch.DIFFICULTIES)
    print("Pre-generating Sudoku puzzles... This might take a moment.")

    with tqdm(total=total_puzzles, ncols=100) as pbar:
        for diff in TrainingBatch.DIFFICULTIES:
            PREGENERATED_PUZZLES[diff] = []
            pbar.set_description(f"Generating puzzles [{diff}]")
            for _ in range(NUM_PREGENERATED_PER_DIFFICULTY):
                puzzle, solution = generate_sudoku(diff)
                PREGENERATED_PUZZLES[diff].append((puzzle.flatten(), solution.flatten()))
                pbar.update(1)
    # Save pregenerated puzzle into a file
    np.save('pregenerated_puzzles.npy', PREGENERATED_PUZZLES)
    print(f"✅ Finished pre-generating {total_puzzles} puzzles across {len(TrainingBatch.DIFFICULTIES)} difficulties.")

def sudoku_loss(model, hidden_states, board_inputs, board_targets, segments):
    config = model.config

    (next_hidden_states, output_logits,
     q_act_halt_logits, q_act_continue_logits) = model(hidden_states, board_inputs)

    # Output loss for Sudoku prediction
    output_loss = F.cross_entropy(
        output_logits.view(-1, config.vocab_size),
        board_targets.view(-1),
        reduction='none'
    ).view(board_inputs.shape)

    output_loss_mask = (board_inputs == 0).float()
    masked_output_loss = (output_loss * output_loss_mask).sum() / output_loss_mask.sum().clamp(min=1)

    # Accuracy for halting decision target
    with torch.no_grad():
        predictions = output_logits.argmax(dim=2)
        output_accuracy = ((predictions == board_targets) | (board_inputs != 0)).all(dim=1).long()

    # Q-ACT loss
    next_segments = segments + 1
    is_last_segment = next_segments >= config.act.halt_max_steps

    is_halted = is_last_segment | (q_act_halt_logits > q_act_continue_logits)

    # Halt exploration logic
    halt_exploration = torch.rand_like(q_act_halt_logits) < config.act.halt_exploration_probability
    if halt_exploration.any():
        min_halt_segments = torch.randint(2, config.act.halt_max_steps + 1, segments.shape, device=segments.device)
        min_halt_segments = min_halt_segments.long() * halt_exploration.long()
        is_halted = is_halted & (next_segments > min_halt_segments)

    # Compute target for Q-Continue
    with torch.no_grad():
        (_, _, next_q_act_halt, next_q_act_continue) = model(next_hidden_states, board_inputs)

    q_act_continue_target = torch.where(
        is_last_segment,
        next_q_act_halt,
        torch.maximum(next_q_act_halt, next_q_act_continue)
    ).sigmoid()

    q_act_loss = (
        F.binary_cross_entropy_with_logits(q_act_halt_logits, output_accuracy.float(), reduction='none') + \
        F.binary_cross_entropy_with_logits(q_act_continue_logits, q_act_continue_target, reduction='none')
    ) / 2
    avg_q_act_loss = q_act_loss.mean()

    total_loss = masked_output_loss + avg_q_act_loss

    # Metrics for logging
    with torch.no_grad():
        full_accuracy = ((predictions == board_targets) | (board_inputs != 0)).float().mean()
        q_act_halt_accuracy = ((q_act_halt_logits >= 0) == output_accuracy.bool()).float().mean()

    return (total_loss, masked_output_loss, avg_q_act_loss, is_halted,
            full_accuracy, q_act_halt_accuracy, next_hidden_states)

class TrainingBatch:
    DIFFICULTIES = [Difficulty.EASY, Difficulty.MEDIUM, Difficulty.HARD, Difficulty.EXTREME]
    CURRICULUM_PROBAS = [
        [1.0, 0.0, 0.0, 0.0],
        [0.7, 0.3, 0.0, 0.0],
        [0.5, 0.4, 0.1, 0.0],
        [0.3, 0.3, 0.3, 0.1],
        [0.1, 0.3, 0.4, 0.2],
    ]

    def __init__(self, model: HRMACTInner, batch_size: int, device: torch.device):
        self.model = model
        self.batch_size = batch_size
        self.device = device
        self.curriculum_level = 0
        self.total_puzzles = 0

        # Ensure puzzles are pre-generated
        if not PREGENERATED_PUZZLES:
            _generate_puzzle_pool()

        # Keep track of which puzzles in the pool have been used
        self.puzzle_pool_indices = {diff: list(range(len(PREGENERATED_PUZZLES[diff]))) for diff in TrainingBatch.DIFFICULTIES}
        self.current_puzzle_indices_ptr = {diff: 0 for diff in TrainingBatch.DIFFICULTIES}

        hidden_size = model.config.transformers.hidden_size
        seq_len = model.config.seq_len

        self.board_inputs = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        self.board_targets = torch.zeros((batch_size, seq_len), dtype=torch.long, device=device)
        self.segments = torch.zeros(batch_size, dtype=torch.long, device=device)

        # FIX 1: Initialize hidden_states with new, independent tensors, not views of model parameters.
        low_level_h = torch.zeros((batch_size, seq_len + 1, hidden_size), dtype=model.config.dtype, device=device)
        high_level_h = torch.zeros((batch_size, seq_len + 1, hidden_size), dtype=model.config.dtype, device=device)
        self.hidden_states = (low_level_h, high_level_h)

        for i in range(batch_size):
            self.replace(i)

    def _sample_difficulty(self):
        return random.choices(self.DIFFICULTIES, self.CURRICULUM_PROBAS[self.curriculum_level], k=1)[0]

    def _sample_puzzle_from_pool(self, difficulty: Difficulty):
        # Sample without replacement for a cleaner batch (or with replacement for infinite stream)
        # For simplicity, let's just cycle through for now.
        pool = PREGENERATED_PUZZLES[difficulty]
        pool_len = len(pool)

        ptr = self.current_puzzle_indices_ptr[difficulty]
        puzzle_flat, solution_flat = pool[ptr]

        self.current_puzzle_indices_ptr[difficulty] = (ptr + 1) % pool_len

        # Shuffle the pool if we've cycled through to keep things random
        if self.current_puzzle_indices_ptr[difficulty] == 0:
            random.shuffle(PREGENERATED_PUZZLES[difficulty]) # Shuffle the actual list of tuples

        return puzzle_flat, solution_flat

    def replace(self, idx: int):
        # Instead of generate_sudoku, sample from pre-generated pool
        puzzle_flat, solution_flat = self._sample_puzzle_from_pool(self._sample_difficulty())

        self.board_inputs[idx] = torch.tensor(puzzle_flat, device=self.device)
        self.board_targets[idx] = torch.tensor(solution_flat, device=self.device)

        with torch.no_grad():
            self.segments[idx] = 0
            seq_len = self.model.config.seq_len
            low_level_h, high_level_h = self.hidden_states
            low_level_h[idx] = self.model.initial_low_level.unsqueeze(0).expand(seq_len + 1, -1)
            high_level_h[idx] = self.model.initial_high_level.unsqueeze(0).expand(seq_len + 1, -1)

        self.total_puzzles += 1

    def graduate(self):
        if self.curriculum_level + 1 < len(self.CURRICULUM_PROBAS):
            self.curriculum_level += 1
            print(f"Graduated to curriculum level {self.curriculum_level}.")
        else:
            print("Reached highest curriculum level.")

def train_step(model, optimizer, batch):
    optimizer.zero_grad()

    (loss, out_loss, q_loss, is_halted,
     out_acc, q_acc, next_h) = sudoku_loss(
        model,
        batch.hidden_states,
        batch.board_inputs,
        batch.board_targets,
        batch.segments
    )

    loss.backward()
    optimizer.step()

    print(
        f"Output [Loss: {out_loss.item():.4f}, Acc: {out_acc.item():.4f}] | "
        f"Q-ACT [Loss: {q_loss.item():.4f}, Acc: {q_acc.item():.4f}] | "
        f"Puzzles [{batch.total_puzzles}] | Curriculum [{batch.curriculum_level}]"
    )

    batch.hidden_states = next_h
    batch.segments += 1

    halted_indices = torch.where(is_halted)[0]
    for idx in halted_indices:
        batch.replace(idx.item())

    return out_acc.item()
