import torch
import torch.nn as nn

class HandwritingFeatureExtractor(nn.Module):
    """
    The Base CNN: This acts as the 'twin'. 
    It takes a 224x224 grayscale image and shrinks it into a 128-number feature vector.
    """
    def __init__(self):
        super(HandwritingFeatureExtractor, self).__init__()
        
        # Convolutional layers to find edges, curves, and tremors
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5 #create a 5*5 pixel window
            , stride=1 #shift 1 window at a time
            , padding=2 # add 2 pixel padding around image to reducese the detailse loss
            ),
            # rectifier linear unit which make -ev to +ev
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # Image becomes 112x112
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), # Image becomes 56x56
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # Image becomes 28x28
        )
        
        # Fully connected layers to flatten the image into mathematical coordinates
        self.fc = nn.Sequential(
            nn.Linear(128 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5), # Prevents the model from memorizing specific patients
            nn.Linear(512, 128) # Final output: a 128-dimensional embedding
        )

    def forward(self, x):
        output = self.cnn(x)
        output = output.view(output.size()[0], -1) # Flatten the 2D matrices
        output = self.fc(output)
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
        # Pass Image A through the network to get its coordinates
        embedding_a = self.twin_cnn(image_yesterday)
        
        # Pass Image B through the SAME network to get its coordinates
        embedding_b = self.twin_cnn(image_today)
        
        # Return both sets of coordinates so the Loss Function can compare them
        return embedding_a, embedding_b