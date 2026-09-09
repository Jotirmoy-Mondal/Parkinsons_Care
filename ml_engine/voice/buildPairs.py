import os
import random
# -------------------------------------------------------------------------
# EXPLANATION FOR IMPORTS:
# This block brings in standard Python libraries. 
# 'os' allows Python to read your file system (folders and file names).
# 'random' provides the mathematical tools to shuffle data and pick random items.
# -------------------------------------------------------------------------


def list_wav_files(directory: str) -> list:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".wav")]
# -------------------------------------------------------------------------
# EXPLANATION FOR list_wav_files:
# Takes: `directory` (string) — the name of the folder you want to search.
# Returns: A list of strings — the full file paths of every .wav file found.
# What it does: It looks inside the specified folder, ignores anything that 
# isn't an audio file, and stitches the folder path and file name together 
# so Python knows exactly where the file lives (e.g., "ml_engine/data/file.wav").
# -------------------------------------------------------------------------


def split_files(files: list, val_ratio: float = 0.2, seed: int = 42) -> tuple:
    """Split raw files into train/val BEFORE generating pairs — avoids the
    same file appearing in both train and validation pairs."""
    random.Random(seed).shuffle(files)
    split_idx = int(len(files) * (1 - val_ratio))
    return files[:split_idx], files[split_idx:]
# -------------------------------------------------------------------------
# EXPLANATION FOR split_files:
# Takes: 
#   - `files` (list) — the list of audio file paths.
#   - `val_ratio` (float) — the percentage of files to hold back for testing (default 20%).
#   - `seed` (int) — a fixed number to ensure the random shuffle is identical every time.
# Returns: A tuple containing two lists (train_files, validation_files).
# What it does: It scrambles the list of files, calculates exactly where the 
# 80% mark is (`split_idx`), and slices the list into two separate piles. 
# Doing this BEFORE making pairs guarantees zero data leakage.
# -------------------------------------------------------------------------


def generate_pairs(healthy_files: list, pd_files: list, num_pairs: int) -> list:
    pairs = []
    for _ in range(num_pairs // 2):
        a, b = random.sample(healthy_files, 2)
        pairs.append((a, b, 1))  # similar: both healthy

        h = random.choice(healthy_files)
        p = random.choice(pd_files)
        pairs.append((h, p, 0))  # dissimilar: healthy vs PD

    random.shuffle(pairs)
    return pairs
# -------------------------------------------------------------------------
# EXPLANATION FOR generate_pairs:
# Takes: 
#   - `healthy_files` (list) — paths to voices without Parkinson's.
#   - `pd_files` (list) — paths to voices with Parkinson's.
#   - `num_pairs` (int) — the total number of audio pairs you want to create.
# Returns: A list of tuples. Each tuple contains (file_1_path, file_2_path, label).
# What it does: This is the core logic for a Siamese Neural Network. It loops 
# to create data in pairs. Half the time it pairs two healthy voices together 
# and labels them '1' (meaning "these are the same class"). The other half, 
# it pairs a healthy voice with a PD voice and labels them '0' ("different class"). 
# Finally, it shuffles the pairs so the model doesn't learn in a predictable pattern.
# -------------------------------------------------------------------------


def build_train_val_pairs(
    healthy_dir: str, pd_dir: str,
    train_pairs_count: int = 800, val_pairs_count: int = 200,
    val_ratio: float = 0.2,
) -> tuple:
    healthy_files = list_wav_files(healthy_dir)
    pd_files = list_wav_files(pd_dir)

    healthy_train, healthy_val = split_files(healthy_files, val_ratio)
    pd_train, pd_val = split_files(pd_files, val_ratio)

    print(f"Healthy: {len(healthy_train)} train files, {len(healthy_val)} val files")
    print(f"PD: {len(pd_train)} train files, {len(pd_val)} val files")

    train_pairs = generate_pairs(healthy_train, pd_train, train_pairs_count)
    val_pairs = generate_pairs(healthy_val, pd_val, val_pairs_count)

    return train_pairs, val_pairs
# -------------------------------------------------------------------------
# EXPLANATION FOR build_train_val_pairs:
# Takes: 
#   - Folder paths for healthy and PD directories.
#   - The total number of training and validation pairs to generate.
#   - The split ratio (default 20%).
# Returns: A tuple containing two lists (train_pairs, validation_pairs).
# What it does: This is the "orchestrator" function. It calls all the smaller 
# functions above in the correct order:
#   1. Grabs all the raw files.
#   2. Splits them into train/val piles strictly.
#   3. Prints a sanity-check to the terminal so you know how many files you have.
#   4. Generates the actual paired datasets ready for the PyTorch model.
# -------------------------------------------------------------------------


if __name__ == "__main__":
    train_pairs, val_pairs = build_train_val_pairs(
        healthy_dir="ml_engine/data/raw/healthy_voice",
        pd_dir="ml_engine/data/raw/parkinsons_voice",
    )
    print(f"Train pairs: {len(train_pairs)}, Val pairs: {len(val_pairs)}")
# -------------------------------------------------------------------------
# EXPLANATION FOR if __name__ == "__main__":
# Takes/Returns: Nothing directly.
# What it does: This block only runs if you execute this specific file directly 
# from the terminal (e.g., `python -m ml_engine.voice.build_pairs`). It feeds 
# your actual project folders into the orchestrator function and prints the 
# final pair counts to prove the script works. If another file imports this 
# code, this bottom block is ignored.
# -------------------------------------------------------------------------