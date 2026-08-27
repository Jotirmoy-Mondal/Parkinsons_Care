# ml_engine/voice/train.py



"""The Training Execution Flowchart
[ 📥 STEP 1: Data Preparation ]

The system takes your list of paired audio files (e.g., Patient A vs. Healthy B).

It converts the raw audio into visual heatmaps (Mel-spectrograms).
⬇️

[ 📦 STEP 2: Batching & Padding (The collate_fn) ]

Because people speak at different speeds, the spectrograms are all different lengths.

The system finds the longest audio clip in the current batch and instantly pads the shorter clips with empty space so they all fit into a perfect, uniform 3D box.
⬇️

[ 🧠 STEP 3: Engine Initialization ]

The empty AI framework (SiameseVoiceNet) wakes up.

The grading rubric (ContrastiveLoss) is set to pull similar voices together and push different voices apart.

The learning engine (Adam Optimizer) prepares to track the math.
⬇️

[ 🔄 STEP 4: The Core Learning Loop ] (Repeats for 20 Epochs)

Extract: The twin networks look at the batch of spectrograms and extract their 128-number voice fingerprints.

Grade: The Contrastive Loss calculates the exact mathematical distance between the fingerprints and grades how right or wrong the AI was.

Learn: The calculus engine (backpropagation) updates the AI's brain connections to make it smarter for the next batch.
⬇️

[ 💾 STEP 5: Graduation & Saving ]

Once all 20 epochs finish, the AI's learning engine is turned off.

The fully trained mathematical memories are saved to your hard drive as voice_model.pth."""

import torch
import torch.nn as nn #nuralnetwork
# Dataset acts as the blueprint to load and pair our patient files.
# DataLoader acts as the delivery engine to shuffle and batch those files for the AI.
from torch.utils.data import Dataset, DataLoader

from ml_engine.voice.model import SiameseVoiceNet
from ml_engine.voice.preprocess import preprocess_audio


class ContrastiveLoss(nn.Module):
    

    """
    Standard Siamese contrastive loss:
    - Similar pairs (label=1): pulls embeddings closer, penalizes large distance
    - Dissimilar pairs (label=0): pushes embeddings apart, penalizes distance
    being smaller than `margin` (no penalty once far enough apart)
    """
    def __init__(self, margin: float = 1.0):
        
        
        super().__init__()
        self.margin = margin

    def forward(self, emb1, emb2, label):
          
          
          
          # INPUT: The coordinate differences for a whole batch of image pairs.
          # MATH: Calculates the standard L2 norm(p=2) Euclidean distance strictly across the 128 dimensions (dim=1).
          # OUTPUT: Returns a 1D tensor of distances (kept inside PyTorch so the optimizer can track gradients).
         distance = torch.norm(emb1 - emb2, p=2, dim=1)

          # If the pair is similar (label=1), calculate the squared distance.
          # This mathematically forces the AI to pull matching features closer together.
          # If the pair is dissimilar (label=0), this line safely zeroes out.
         loss_similar = label * distance.pow(2)


          # For dissimilar pairs (label=0), this activates.
          # If the distance is smaller than our margin (1.0), it applies a squared penalty to push them apart.
          # If they are already further apart than the margin, the penalty is safely clamped to 0. 
          #The torch.clamp() function restricts (or "clamps") a set of numbers so they cannot go past a certain limit.
         loss_dissimilar = (1 - label) * torch.clamp(self.margin - distance, min=0).pow(2)

          # Averages the total penalty across the entire batch to hand back to the optimizer.
         return (loss_similar + loss_dissimilar).mean()


class VoicePairDataset(Dataset):
    """
    Expects a list of (audio_path_1, audio_path_2, label) tuples.
    label: 1 = similar pair (both healthy, or same patient/stable),
     0 = dissimilar pair (healthy vs PD, or stable vs declined)
    """
    def __init__(self, pairs: list):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        path1, path2, label = self.pairs[idx]
        mel1 = preprocess_audio(path1)
        mel2 = preprocess_audio(path2)
        return mel1, mel2, torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    """
    Mel-spectrograms have variable time length (different recording durations),
    so we can't just torch.stack them directly — pad the shorter ones to match
    the longest in this batch.
    """

    #m.shape hold (e.g. [1, 128, 400])
    #  "*" for unpack the batch
    mels1, mels2, labels = zip(*batch)

    #Dynamic Sequence Length: Extract the maximum temporal dimension (axis -1) across all tensors 
    #to establish the dynamic padding boundary for the current batch.
    max_len = max(m.shape[-1] for m in mels1 + mels2)

    def pad(mel):
        pad_amount = max_len - mel.shape[-1]

        #The first number (0) tells PyTorch to add absolutely nothing to the beginning (left side) of the audio. The second number (pad_amount) tells it to add that exact number of empty zeros to the very end (right side) of the audio.

        #It returns a brand new PyTorch Tensor. The actual vocal audio remains completely unchanged, but the physical length of the tensor is now perfectly stretched to match the longest clip in the batch.
        return torch.nn.functional.pad(mel, (0, pad_amount))

     #each pad(m) return pytorch tensor create a list
    mels1 = torch.stack([pad(m) for m in mels1])
    mels2 = torch.stack([pad(m) for m in mels2])
    labels = torch.stack(labels)
    return mels1, mels2, labels # return list of same size melspectogram data


def train(pairs: list, epochs: int = 20, batch_size: int = 8, lr: float = 1e-4): # lr is learning rate: It controls how big a step the optimizer takes when adjusting the model's weights after each batch, based on the calculated loss/error.1e-4   ==  0.0001
    dataset = VoicePairDataset(pairs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    model = SiameseVoiceNet(embedding_dim=128)
    model.train()

    criterion = ContrastiveLoss(margin=1.0) #measures how wrong the model is
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Adam is a popular choice (adaptive learning rate; adjusts step size per parameter).

    for epoch in range(epochs):
        total_loss = 0.0
        for mel1, mel2, label in loader:
            optimizer.zero_grad()
            emb1, emb2 = model(mel1, mel2)
            loss = criterion(emb1, emb2, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), "ml_engine/voice/weights/voice_model.pth")
    print("Model saved.")


if __name__ == "__main__":
    # Replace with your actual dataset pairs
    pairs = [
        ("data/healthy_1.wav", "data/healthy_2.wav", 1),
        ("data/healthy_1.wav", "data/pd_1.wav", 0),
        # ... build this list from your dataset
    ]
    train(pairs)