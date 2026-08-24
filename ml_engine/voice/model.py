# ml_engine/voice/model.py
"""
### Part 1: The Fingerprint Machine (Siamese Encoder)

This is the journey of a single voice recording (Mel-Spectrogram) passing through the AI.

**[ 📥 START: The Acoustic Picture ]**
The AI receives a visual heatmap of the patient's voice (the Mel-Spectrogram).
↓
**[ 🔎 Feature Extractor Block 1 ]**
The AI scans the image using mathematical filters. It looks for simple, low-level audio patterns (like loud edges or sudden drops). It then shrinks the image size by half to focus on the big picture.
↓
**[ 🔎 Feature Extractor Block 2 ]**
The AI scans the shrunken image again with a new set of filters. It looks for more complex patterns (like the shape of specific vowels). It shrinks the image by half again.
↓
**[ 🔎 Feature Extractor Block 3 ]**
The AI scans the image one last time, looking for high-level, disease-specific features (like microscopic vocal cord stutters or tremors). It shrinks the image by half one final time.
↓
**[ 📏 The Equalizer (Global Adaptive Pooling) ]**
*This is the smartest part of the network.* If a patient speaks for 3 seconds, their acoustic picture is short. If they speak for 10 seconds, their picture is long. This step automatically squashes the picture down to a fixed `1x1` size, regardless of how long the original recording was. This means you don't have to chop up or stretch the patient's audio!
↓
**[ 🔨 The Flattener ]**
The 3D block of extracted audio features is literally smashed flat into a single, straight 1D line of numbers.
↓
**[ 💎 The Embedding Head (Linear Layer) ]**
This straight line of numbers is passed through one final filter that compresses the data down into exactly **128 numbers**.
↓
**[ 📤 OUTPUT: The Digital Fingerprint (Embedding) ]**
The AI spits out a 128-number "fingerprint" that perfectly represents the unique acoustic signature of that patient's voice on that specific day.

---

### Part 2: The Comparison Engine (Siamese Voice Net)

This explains how the "Twin" network tracks disease progression by comparing two files.

**[ 📥 INPUT A ]** Patient's Baseline Voice (Recorded 6 months ago)
*...and simultaneously...*
**[ 📥 INPUT B ]** Patient's Current Voice (Recorded today)
↓
**[ 🤖 THE CLONED BRAIN ]**
Both Input A and Input B are pushed through the **exact same** Fingerprint Machine (The Encoder) described in Part 1. Because it is the exact same brain looking at both files, it judges them using the exact same rules.
↓
**[ 📤 OUTPUT A ]**
A 128-number fingerprint of the Baseline voice.
*...and simultaneously...*
**[ 📤 OUTPUT B ]**
A 128-number fingerprint of Today's voice.
↓
**[ 🏁 FINISH ]**
The AI hands both fingerprints back to your main system. Because they are just lists of numbers, your system can now easily calculate the mathematical "distance" between them.

* If the distance is small -> The voice hasn't changed -> **Stability Score is High.**
* If the distance is large -> The vocal tremors have worsened -> **Stability Score is Low.**"""

import torch
import torch.nn as nn


class SiameseEncoder(nn.Module):
    """
    Generic conv encoder — same architecture can serve both the voice
    (mel-spectrogram) and handwriting (drawing image) modalities, just
    instantiated with different in_channels/input shapes.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128):
        super().__init__()


        # The Conv Stack: A 3-step assembly line that progressively scans the audio 
        # and self used for keep it in memory and forward fn can find that later




        """self.conv_stack = nn.Sequential(
        nn.Conv2d(...),   # Station 1: The Scanner
        nn.BatchNorm2d(), # Station 2: The Stabilizer
        nn.ReLU(),        # Station 3: The Rule Maker
        nn.MaxPool2d(2)   # Station 4: The Shrinker
        )"""
        # nn.Sequential acts as a conveyor belt. It takes a comma-separated list 
        # of PyTorch layers (like Conv2d and ReLU) and automatically pushes the 
        # data through them in the exact top-to-bottom order they are written here.
        
        self.conv_stack = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            
            # nn.BatchNorm2d(num_features)
            # Takes: A 4D tensor with 16 channels(equal of outer channel).
            # Returns: A 4D tensor of the exact same size/shape.
            # How: It mathematically "squashes" wildly large or small numbers back into 
            # a stable, healthy range (centered around 0). This acts as an auto-leveler, 
            # preventing the AI's math from exploding and helping it learn much faster.
            nn.BatchNorm2d(16),


            # nn.ReLU()
            # Takes: A 4D tensor.
            # Returns: A 4D tensor of the exact same size/shape.
            # How: Applies the Non-Linear rule: max(0, x). Any negative numbers are instantly 
            # turned to zero; positive numbers are left alone. This deliberate "bending" of 
            # the math prevents the network from only learning straight lines, allowing it to 
            # learn the highly complex, curved, and irregular patterns of human disease.
            nn.ReLU(),


            # The Shrinker: Scans the picture in 2x2 chunks, keeping only the highest number 
            # and throwing the rest away. This cuts the image size in half and distills the 
            # data down to only the most dominant acoustic features (ignoring background noise).
            nn.MaxPool2d(2),

            # nn.Conv2d(in_channels, out_channels, kernel_size, padding)
            # Takes: A 4D tensor (e.g., [1 batch, 1 channel, 80 mels, time_frames])
            # Returns: A 4D tensor (e.g., [1 batch, 16 channels, 80 mels, time_frames])
            # How: It slides a 3x3 mathematical filter over the audio picture, looking 
            # for specific patterns, and outputs 16 new deep feature channels.

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Collapses variable time/frequency dims to a fixed 1x1,
        # so recordings of different lengths all produce the same
        # size output without needing padding/cropping.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Takes: A 4D tensor of ANY variable height and width (e.g., [1, 64, H, W]).
        # Returns: A fixed 4D tensor of size [1, 64, 1, 1].
        # How: Patients record audio for different lengths of time, resulting in 
        # spectrograms of different sizes. This layer takes the mathematical average 
        # of an entire feature map and squashes it into a single 1x1 pixel. This 
        # safely erases the "Time" dimension, ensuring the network always outputs a 
        # fixed-size tensor regardless of whether the audio is 2 seconds or 10 seconds.



        self.embedding_head = nn.Linear(64, embedding_dim)


        # self.embedding_head = nn.Linear(64, embedding_dim)
        # Takes: A flat 1D tensor of 64 extracted acoustic features.
        # Returns: A flat 1D tensor of 128 numbers (the embedding).
        # How: This is a Fully Connected layer. It acts as the "Head" of the network, 
        # taking the raw medical features and mathematically projecting them into a 
        # 128-dimensional coordinate. This specific coordinate is the unique "Digital 
        # Fingerprint" of the patient's voice, which the Siamese network will use to 
        # compare against past recordings.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, in_channels, height, width]
        x = self.conv_stack(x)
        x = self.global_pool(x)
        x = x.flatten(1) 
        
        # x = x.flatten(1)
        # Takes: A 4D tensor shaped [batch, 64, 1, 1].
        # Returns: A 2D tensor shaped [batch, 64].
        # How: The upcoming Linear layer only accepts flat, straight lines of data. 
        # This command smashes the remaining spatial dimensions into a single flat line 
        # of 64 numbers. The '1' tells PyTorch to preserve Dimension 0 (the Batch size) 
        # so that if multiple patients are processed at once, their data isn't mixed together.

        embedding = self.embedding_head(x)  # [batch, embedding_dim]
        return embedding


class SiameseVoiceNet(nn.Module):
    """
    Twin-network wrapper: runs the same encoder on two inputs
    (e.g., today's sample vs. anchor) and returns both embeddings.
    """

    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = SiameseEncoder(in_channels=1, embedding_dim=embedding_dim)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        emb1 = self.encoder(x1)
        emb2 = self.encoder(x2)
        return emb1, emb2



    # def embed(self, x: torch.Tensor) -> torch.Tensor:
    # Takes: A single 4D audio tensor (e.g., just Today's recording).
    # Returns: A single 1D tensor (the 128-number digital fingerprint).
    # How: While forward() forces a two-file comparison, this helper function 
    # pushes a single file through the encoder. This is highly useful for Day 1 
    # of the app, allowing you to generate the patient's Baseline "Anchor" 
    # fingerprint and save it to the database for future comparisons.
    
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Get a single embedding, e.g. for inference against a stored anchor."""
        return self.encoder(x)