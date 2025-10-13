import subprocess
import sys
import argparse
import os

MAIN_SCRIPT_PATH = 'main.py'

def run_inference(checkpoint_path: str, difficulty: str, turns: int, verbose: bool):
    """Runs inference with the specified parameters."""
    if not os.path.exists(MAIN_SCRIPT_PATH):
        print(f"Error: Could not find '{MAIN_SCRIPT_PATH}'")
        sys.exit(1)
    if not os.path.exists(checkpoint_path):
        print(f"Error: Could not find checkpoint file '{checkpoint_path}'")
        sys.exit(1)

    command = [
        sys.executable, MAIN_SCRIPT_PATH,
        "infer", checkpoint_path,
        difficulty,
        "--turns", str(turns),
    ]
    if verbose:
        command.append("--verbose")

    print("Running command:", " ".join(command))
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with return code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run inference with specified parameters."
    )
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint (e.g., model.pth)")
    parser.add_argument("difficulty", type=str, help="Difficulty level (e.g., extreme)")
    parser.add_argument("--turns", type=int, default=1, help="Number of turns for inference")
    parser.add_argument("--verbose", type=bool, default=False, help="Number of turns for inference")

    args = parser.parse_args()
    run_inference(args.checkpoint, args.difficulty, args.turns, args.verbose)
