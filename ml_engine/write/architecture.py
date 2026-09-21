"""
=============================================================================
LOGIC FLOWCHART: Siamese Parkinson's Handwriting Network
=============================================================================
Input Image A (e.g., Yesterday)                 Input Image B (e.g., Today)
      | (224x224 grayscale)                           | (224x224 grayscale)
      v                                               v
+---------------------------------------------------------------------------+
|                    HandwritingFeatureExtractor (Twin CNN)                 |
|                           (Shared Weights)                                |
+---------------------------------------------------------------------------+
      |                                               |
      |--> cnn (Conv2d, BatchNorm, ReLU, MaxPool) <---|
      |    (Reduces spatial dimensions & extracts features)
      v                                               v
      |--> gap (AdaptiveAvgPool2d) <------------------|
      |    (Averages spatial dims to 1x1)
      v                                               v
      |--> fc (Linear, ReLU, Dropout, Linear) <-------|
      |    (Maps features to a 128-dimensional space)
      v                                               v
      |--> normalize (L2 Normalization) <-------------|
      |    (Scales vector to have a length of 1)
      v                                               v
128-dim Embedding A                             128-dim Embedding B
      |                                               |
      +----------------------> <----------------------+
                         (Loss Function compares them)
=============================================================================
"""

import torch
import torch.nn as nn

class HandwritingFeatureExtractor(nn.Module):
    """
    The Base CNN: This acts as the 'twin'. 
    It takes a 224x224 grayscale image and shrinks it into a 128-number feature vector.
    """
    def __init__(self):
        # super():
        # TAKES: The child class and the instance (self).
        # RETURNS: A proxy object that allows you to call methods of the parent class (nn.Module), initializing the base class.
        super(HandwritingFeatureExtractor, self).__init__()
        
        # nn.Sequential():
        # TAKES: A comma-separated sequence of PyTorch neural network modules.
        # RETURNS: A container module that passes the data through each layer sequentially in the exact order they are defined.
        self.cnn = nn.Sequential(
            # nn.Conv2d():
            # TAKES: in_channels (1 for grayscale), out_channels (32 filters), kernel_size (5x5 grid), stride (1 means move 1 column at a time), and padding (2).
            # RETURNS: A 2D convolutional layer that extracts spatial features like edges and curves.
            nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2),
            
            # nn.BatchNorm2d():
            # TAKES: num_features (32 channels matching the output of the previous Conv2d layer).
            # RETURNS: A layer that normalizes the batch data to have a mean of 0 and variance of 1, speeding up and stabilizing training.
            nn.BatchNorm2d(32),
            
            # nn.ReLU():
            # TAKES: inplace boolean (if True, it modifies the data directly in memory rather than creating a copy).
            # RETURNS: An activation function layer that replaces all negative numbers with zero, introducing non-linearity.
            nn.ReLU(inplace=True),
            
            # nn.MaxPool2d():
            # TAKES: kernel_size (2) and stride (2).
            # RETURNS: A pooling layer that shrinks the image dimensions by half by keeping only the maximum values in each 2x2 region.
            nn.MaxPool2d(2, 2), # Image becomes 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # Image becomes 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Image becomes 28x28
        )

        # nn.AdaptiveAvgPool2d():
        #For every single one of those 128 feature maps, it takes all the numbers in that specific grid, adds them up, divides by the total, and spits out one single average number.

        #The massive 128 x 28 x 28 block of data is instantly crushed down into a flat list of exactly 128 numbers
        # TAKES: output_size (1 means a 1x1 spatial resolution).
        # RETURNS: A pooling layer that dynamically calculates the necessary window size to average out spatial dimensions, yielding 128x1x1.

        #because nn,linear only take 1d input
        self.gap = nn.AdaptiveAvgPool2d(1) #Global Average Pooling

        self.fc = nn.Sequential(
            # nn.Linear():
            # TAKES: in_features (the total incoming connections) and out_features (the target size for the output).
            # RETURNS: A fully connected linear layer that applies a mathematical matrix multiplication (y = xA^T + b).
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            
            # nn.Dropout():
            # TAKES: Probability 'p' (0.3 means a 30% chance).
            # RETURNS: A regularization layer that randomly zeroes out neurons during training to prevent the network from memorizing the data (overfitting).
            nn.Dropout(p=0.3), 
            nn.Linear(128, 128) 
        )

    def forward(self, x):
        # NOTE: Fixed missing step to pass input 'x' through 'self.cnn' first.
        output = self.cnn(x)
        output = self.gap(output)
        
        # output.size():
        # TAKES: An index indicating the desired dimension (0 targets the batch size dimension).
        # RETURNS: An integer representing the size of that specific dimension (how many images are in the current batch).
        
        # output.view():
        # TAKES: The desired tensor dimensions (batch size, and -1 to automatically infer the remaining flattened size).
        # RETURNS: A new tensor with the exact same data, but reshaped from a multi-dimensional grid into a flat 2D shape (batch_size, features).
        output = output.view(output.size(0), -1) 
        
        output = self.fc(output)
        
        # torch.nn.functional.normalize():
        # TAKES: input tensor, 'p' (the norm degree, 2 means L2/Euclidean norm), and 'dim' (the dimension along which to normalize, 1 means feature dimension).
        # RETURNS: A normalized tensor where each feature vector is scaled to have a length (magnitude) of exactly 1.
        output = torch.nn.functional.normalize(output, p=2, dim=1)
        
        return output


class SiameseParkinsonNetwork(nn.Module):
    """
    The Siamese Wrapper: This holds the two identical twin CNNs.
    """
    def __init__(self):
        super(SiameseParkinsonNetwork, self).__init__()
        # We instantiate the feature extractor ONCE. 
        # Both images will pass through this exact same network (sharing weights).
        self.twin_cnn = HandwritingFeatureExtractor()

    def forward(self, image_yesterday, image_today):
        # Pass Image A through the network to get its 128-dimensional coordinates
        embedding_a = self.twin_cnn(image_yesterday)
        
        # Pass Image B through the SAME network to get its coordinates
        embedding_b = self.twin_cnn(image_today)
        
        # Return both sets of coordinates so the Loss Function (e.g., Contrastive Loss) can compare them
        return embedding_a, embedding_b