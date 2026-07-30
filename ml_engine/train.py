import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import ParkinsonSiameseDataset
from architecture import SiameseParkinsonNetwork

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

# Notice we now pass 'test_type' directly into the function
def train_model(test_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Training Initialization ---")
    print(f"Device: {device}")
    print(f"Targeting Dataset: {test_type.upper()}")
    
    image_dir_path = "data/raw/"
    weights_dir = "weights/"

    # The dataset dynamically uses whatever folder you typed in the terminal
    siamese_dataset = ParkinsonSiameseDataset(
        img_dir=image_dir_path,
        subfolder=test_type  
    )
    
    train_dataloader = DataLoader(siamese_dataset, shuffle=True, num_workers=0, batch_size=32)

    net = SiameseParkinsonNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(net.parameters(), lr=0.0005)

    epochs = 20
    
    print("Starting training phase...")
    net.train() 
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)
            optimizer.zero_grad()
            
            output1, output2 = net(img0, img1)
            
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            
            epoch_loss += loss_contrastive.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {epoch_loss / len(train_dataloader):.4f}")

    os.makedirs(weights_dir, exist_ok=True)
    
    # Dynamically names the output file based on your input!
    save_path = os.path.join(weights_dir, f"siamese_{test_type}.pth") 
    torch.save(net.state_dict(), save_path)
    print(f"Success! Weights saved to: {save_path}\n")

if __name__ == "__main__":
    # This captures what you type in the terminal
    parser = argparse.ArgumentParser(description="Train Parkinson's Siamese Network")
    parser.add_argument(
        '--type', 
        type=str, 
        required=True, 
        help="The name of the subfolder to train (e.g., spiral, meander, circle)"
    )
    args = parser.parse_args()
    
    # Pass the terminal argument into the training function
    train_model(args.type)