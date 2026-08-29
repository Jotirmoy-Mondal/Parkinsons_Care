# ml_engine/voice/build_feature_csv.py

import os
import csv
from ml_engine.voice.features import extract_acoustic_features

def build_csv(healthy_dir: str, pd_dir: str, output_csv: str):
    rows = []

    for filename in os.listdir(healthy_dir):
        if filename.endswith(".wav"):
            path = os.path.join(healthy_dir, filename)
            features = extract_acoustic_features(path)
            features["filename"] = filename
            features["status"] = 0  # healthy
            rows.append(features)

    for filename in os.listdir(pd_dir):
        if filename.endswith(".wav"):
            path = os.path.join(pd_dir, filename)
            features = extract_acoustic_features(path)
            features["filename"] = filename
            features["status"] = 1  # Parkinson's
            rows.append(features)

    if not rows:
        print("No WAV files found — check your folder paths.")
        return

    fieldnames = list(rows[0].keys())
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_csv}")


if __name__ == "__main__":
    build_csv(
        healthy_dir="ml_engine/data/raw/healthy_voice",
        pd_dir="ml_engine/data/raw/parkinsons_voice",
        output_csv="ml_engine/data/voice_features.csv",
    )