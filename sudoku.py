# sudoku.py

import random
import numpy as np
from enum import Enum
from datasets import load_dataset
import multiprocessing

class Difficulty(Enum):
    VERY_EASY = (46, 50)
    EASY = (40, 45)
    MEDIUM = (32, 39)
    HARD = (28, 31)
    EXTREME = (17, 27)

def _get_masks(grid):
    rows = [0] * 9
    cols = [0] * 9
    boxes = [0] * 9
    for r in range(9):
        for c in range(9):
            if grid[r, c] != 0:
                val_bit = 1 << (grid[r, c] - 1)
                rows[r] |= val_bit
                cols[c] |= val_bit
                box_idx = (r // 3) * 3 + c // 3
                boxes[box_idx] |= val_bit
    return rows, cols, boxes

def _find_empty(grid):
    for r in range(9):
        for c in range(9):
            if grid[r, c] == 0:
                return r, c
    return None

def _fill_grid_recursive(grid, rows, cols, boxes):
    empty = _find_empty(grid)
    if not empty:
        return True
    r, c = empty

    box_idx = (r // 3) * 3 + c // 3
    used = rows[r] | cols[c] | boxes[box_idx]

    nums = list(range(1, 10))
    random.shuffle(nums)

    for num in nums:
        val_bit = 1 << (num - 1)
        if (used & val_bit) == 0:
            grid[r, c] = num
            rows[r] |= val_bit
            cols[c] |= val_bit
            boxes[box_idx] |= val_bit

            if _fill_grid_recursive(grid, rows, cols, boxes):
                return True

            grid[r, c] = 0
            rows[r] &= ~val_bit
            cols[c] &= ~val_bit
            boxes[box_idx] &= ~val_bit
    return False

def _solve_recursive(grid, solutions, limit, rows, cols, boxes):
    if solutions[0] >= limit:
        return True

    empty = _find_empty(grid)
    if not empty:
        solutions[0] += 1
        return solutions[0] >= limit

    r, c = empty
    box_idx = (r // 3) * 3 + c // 3
    used = rows[r] | cols[c] | boxes[box_idx]

    for num in range(1, 10):
        val_bit = 1 << (num - 1)
        if (used & val_bit) == 0:
            grid[r, c] = num
            rows[r] |= val_bit
            cols[c] |= val_bit
            boxes[box_idx] |= val_bit

            if _solve_recursive(grid, solutions, limit, rows, cols, boxes):
                return True

            grid[r, c] = 0
            rows[r] &= ~val_bit
            cols[c] &= ~val_bit
            boxes[box_idx] &= ~val_bit

    return False

def generate_sudoku(difficulty: Difficulty):
    board = np.zeros((9, 9), dtype=int)
    rows, cols, boxes = _get_masks(board)
    _fill_grid_recursive(board, rows, cols, boxes)

    solution = board.copy()
    puzzle = board.copy()

    target_clues_range = difficulty.value

    cells = list(range(81))
    random.shuffle(cells)

    clues = 81
    cursor = 0

    while cursor < len(cells) and clues > target_clues_range[1]:
        idx = cells[cursor]
        cursor += 1
        r, c = idx // 9, idx % 9

        backup = puzzle[r, c]
        puzzle[r, c] = 0

        test_puzzle = puzzle.copy()
        solutions = [0]

        test_rows, test_cols, test_boxes = _get_masks(test_puzzle)
        _solve_recursive(test_puzzle, solutions, 2, test_rows, test_cols, test_boxes)

        if solutions[0] != 1:
            puzzle[r, c] = backup
        else:
            clues -= 1

    if clues > target_clues_range[0]:
        for j in range(cursor, len(cells)):
            if clues <= target_clues_range[0]:
                break
            idx = cells[j]
            r, c = idx // 9, idx % 9

            if puzzle[r,c] == 0:
                continue

            backup = puzzle[r, c]
            puzzle[r, c] = 0

            test_puzzle = puzzle.copy()
            solutions = [0]

            test_rows, test_cols, test_boxes = _get_masks(test_puzzle)
            _solve_recursive(test_puzzle, solutions, 2, test_rows, test_cols, test_boxes)

            if solutions[0] != 1:
                puzzle[r, c] = backup
            else:
                clues -= 1

    return puzzle, solution

def load_online_puzzle(shard: str, batch_size: int = 1_000, num_proc: int = None) -> dict:
    """
    Load a Sudoku dataset from an online source and group puzzles by their 'missing' count.

    Returns:
        grouped_data: dict where keys are missing counts (as strings)
                      and values are lists of samples (dicts).
                      Each sample includes:
                        - 'puzzle': np.ndarray (9, 9)
                        - 'solution': np.ndarray (9, 9)
                        - 'missing', 'difficulty', 'set', 'solving_time'
    """
    total_cores = multiprocessing.cpu_count()
    num_proc = max(1, (num_proc or total_cores - 1))

    # Load the dataset from Hugging Face
    print("[INFO] Loading dataset...")
    ds = load_dataset("Ritvik19/Sudoku-Dataset", split=shard)
    print("[INFO] Dataset remove columns.")
    ds = ds.select_columns(['puzzle', 'solution', 'missing'])
    def convert_batch(batch):
        puzzles, solutions = [], []
        for p_str, s_str in zip(batch["puzzle"], batch["solution"]):
            puzzles.append(np.fromiter(map(int, p_str), dtype=np.int64).reshape(9, 9))
            solutions.append(np.fromiter(map(int, s_str), dtype=np.int64).reshape(9, 9))
        return {"puzzle": puzzles, "solution": solutions}

    # Convert 'puzzle' and 'solution' to 9x9 numpy arrays
    print("[INFO] Dataset format conversion.")
    ds = ds.map(convert_batch, batched=True, batch_size=batch_size, num_proc=num_proc)
    print("[INFO] Dataset set remove coulumns is too easy")
    ds = ds.filter(lambda example: example['missing'] != 1, num_proc=num_proc)
    return ds



def sudoku_board_string(board):
    horizontal_line = "+-------+-------+-------+"
    result = horizontal_line + "\n"
    for i, row in enumerate(board):
        line = "|"
        for j, cell in enumerate(row):
            display_value = "." if cell == 0 else str(int(cell))
            line += f" {display_value}"
            if (j + 1) % 3 == 0:
                line += " |"
        result += line + "\n"
        if (i + 1) % 3 == 0:
            result += horizontal_line + "\n"
    return result.strip()
