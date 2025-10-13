# TODO: Implement puzzle generation logic and other utility functions
# def _generate_puzzle_pool():
#     print("Pre-generating Sudoku puzzles... This might take a moment.")
#     for diff in TrainingBatch.DIFFICULTIES:
#         PREGENERATED_PUZZLES[diff] = []
#         for _ in range(NUM_PREGENERATED_PER_DIFFICULTY):
#             puzzle, solution = generate_sudoku(diff)
#             PREGENERATED_PUZZLES[diff].append((puzzle.flatten(), solution.flatten()))
#     print(f"Finished pre-generating {NUM_PREGENERATED_PER_DIFFICULTY * len(TrainingBatch.DIFFICULTIES)} puzzles.")
