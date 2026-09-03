# =========================================================================
# OVERALL FILE EXPLANATION:
# This script is the main "training engine" for your Parkinson's detection AI. 
# It takes the pairs of audio files you generated, feeds them into your PyTorch 
# Siamese Network, measures how wrong the model's predictions are, and updates 
# the model's "brain" (weights) to make it smarter over time. Finally, it saves 
# the smartest version of the model to your hard drive.
# =========================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from ml_engine.voice.model import SiameseVoiceNet
from ml_engine.voice.preprocess import preprocess_audio
from ml_engine.voice.build_pairs import build_train_val_pairs
# -------------------------------------------------------------------------
# EXPLANATION FOR IMPORTS:
# Brings in PyTorch's core neural network tools (`torch.nn`), data loading 
# tools (`Dataset`, `DataLoader`), and the specific custom files you wrote 
# to define the model architecture, process the audio, and build the pairs.
# -------------------------------------------------------------------------


class ContrastiveLoss(nn.Module):
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
        distance = torch.norm(emb1 - emb2, p=2, dim=1)
        loss_similar = label * distance.pow(2)
        loss_dissimilar = (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)
        return (loss_similar + loss_dissimilar).mean()
# -------------------------------------------------------------------------
# EXPLANATION FOR ContrastiveLoss:
# Takes (on Init): `margin` (float) — the minimum distance you want between different classes.
# Takes (on Forward): `emb1`, `emb2` (tensors representing the audio), `label` (1 or 0).
# Returns: A single number (tensor) representing the "error" or "loss" of the batch.
# What it does: This is the penalty system for the AI. If the voices are the 
# same class (label=1), it penalizes the model if their embeddings are far apart. 
# If they are different classes (label=0), it penalizes the model if their 
# embeddings are closer than the `margin`.
# -------------------------------------------------------------------------


class VoicePairDataset(Dataset):
    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        mel1 = preprocess_audio(path1)
        mel2 = preprocess_audio(path2)
        return mel1, mel2, torch.tensor(label, dtype=torch.float32)
# -------------------------------------------------------------------------
# EXPLANATION FOR VoicePairDataset:
# Takes (on Init): `pairs` (list of tuples from your build_pairs script).
# Takes (on GetItem): `idx` (integer) — the specific row number to fetch.
# Returns: `mel1`, `mel2` (processed audio tensors), and `label` (tensor).
# What it does: This acts as a smart conveyor belt. Instead of loading all 
# thousands of audio files into your Codespace's RAM at once (which would crash it), 
# it only loads and processes the specific two audio files needed for the current step.
# -------------------------------------------------------------------------


def collate_fn(batch):
    mels1, mels2, labels = zip(*batch)
    max_len = max(m.shape[-1] for m in mels1 + mels2)

    def pad(mel):
        pad_amount = max_len - mel.shape[-1]
        return torch.nn.functional.pad(mel, (0, pad_amount))

    mels1 = torch.stack([pad(m) for m in mels1])
    mels2 = torch.stack([pad(m) for m in mels2])
    labels = torch.stack(labels)
    return mels1, mels2, labels
# -------------------------------------------------------------------------
# EXPLANATION FOR collate_fn:
# Takes: `batch` (a list of raw outputs from VoicePairDataset).
# Returns: Three neatly stacked Tensors containing the whole batch's data.
# What it does: People speak at different speeds, so your audio files are 
# different lengths. PyTorch cannot do math on jagged, uneven data. This function 
# finds the longest audio clip in the current batch and pads all the shorter 
# clips with empty space (zeros) so they form a perfect, uniform rectangle of data.
# -------------------------------------------------------------------------


def run_epoch(model, loader, criterion, optimizer=None):
    """optimizer=None means evaluation mode — no weight updates."""
    is_training = optimizer is not None
    model.train() if is_training else model.eval()

    total_loss = 0.0
    context = torch.enable_grad() if is_training else torch.no_grad()

    with context:
        for mel1, mel2, label in loader:
            if is_training:
                optimizer.zero_grad()

            emb1, emb2 = model(mel1, mel2)
            loss = criterion(emb1, emb2, label)

            if is_training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / len(loader)
# -------------------------------------------------------------------------
# EXPLANATION FOR run_epoch:
# Takes: `model`, `loader` (data feeder), `criterion` (ContrastiveLoss), 
#        and optionally `optimizer` (the tool that updates the weights).
# Returns: The average loss (float) for this entire cycle.
# What it does: It runs one complete cycle through your data. If you give it 
# an optimizer, it acts as a "training" cycle (updating the model's brain based 
# on its mistakes). If you don't give it an optimizer, it acts as a "testing/validation" 
# cycle (just checking how smart the model is without changing its brain).
# -------------------------------------------------------------------------


def train(train_pairs: list, val_pairs: list, epochs: int = 20, batch_size: int = 8, lr: float = 1e-4):
    train_loader = DataLoader(
        VoicePairDataset(train_pairs), batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        VoicePairDataset(val_pairs), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = SiameseVoiceNet(embedding_dim=128)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")

    for epoch in range(epochs):
        train_loss = run_epoch(model, train_loader, criterion, optimizer)
        val_loss = run_epoch(model, val_loader, criterion, optimizer=None)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

        # Save the checkpoint only when validation improves — protects against
        # overfitting in later epochs (val loss rising while train loss keeps falling)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "ml_engine/voice/weights/voice_model.pth")
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")

    print("Training complete.")
# -------------------------------------------------------------------------
# EXPLANATION FOR train:
# Takes: `train_pairs`, `val_pairs`, `epochs` (cycles), `batch_size`, `lr` (learning rate).
# Returns: Nothing (but it saves a file to your hard drive).
# What it does: This is the orchestrator. It prepares the data loaders, sets 
# up the Adam optimizer, and loops through the `epochs`. Most importantly, it 
# tracks the validation score. It only saves the `.pth` weights file if the 
# model actually performed better on the test data than it did in the last cycle, 
# preventing the AI from just memorizing the training data.
# -------------------------------------------------------------------------


if __name__ == "__main__":
    train_pairs, val_pairs = build_train_val_pairs(
        healthy_dir="ml_engine/data/raw/Healthy_voice",
        pd_dir="ml_engine/data/raw/parkinsons_voice",
        train_pairs_count=800,
        val_pairs_count=200,
    )
    train(train_pairs, val_pairs, epochs=20, batch_size=8)
# -------------------------------------------------------------------------
# EXPLANATION FOR if __name__ == "__main__":
# Takes/Returns: Nothing.
# What it does: The execution block. If you run `python -m ml_engine.voice.train` 
# in the terminal, this block fires. It calls your previous script to get 800 
# train pairs and 200 validation pairs, then kicks off the training loop for 20 cycles.
# -------------------------------------------------------------------------