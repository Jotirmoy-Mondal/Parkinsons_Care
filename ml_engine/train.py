import os
import argparse #for terminal use
import torch
import torch.nn as nn #nural network(nn) layer
import torch.optim as optim
import torch.nn.functional as F
#F contains all the raw mathematical operations and functions that a neural network needs to do its job
from torch.utils.data import DataLoader

from dataset import ParkinsonSiameseDataset
from architecture import SiameseParkinsonNetwork

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__() # to inherit parent constractor
        self.margin = margin
        # here margin is distance between healthy and patient data 


# 1.this fn calculate euclidean distance and constractive loss
    def forward(self, output1, output2, label):
        #p_d funtion take pair of image and calculate Distance = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2 +...} after that keepdim force the result to column.
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True )
        #The constractive loss formula where Y=label,D= euclidean distance and m= slef.margin

        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss_contrastive

        """Contrastive Loss teaches a machine learning model how to understand similarity rather than forcing it to memorize specific categories.

        Instead of asking a model, "Is this a picture of a cat?" (which standard classification does), contrastive loss asks, "Are these two pictures of the same thing?""""

# Notice we now pass 'test_type' directly into the function

#this funciton check device and do dataload and train
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
# net=network
    net = SiameseParkinsonNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = optim.Adam(net.
    parameters(), lr=0.0005)
    #Adam is the "mechanic." When the grader says the AI made a mistake, Adam goes inside the network and turns the mathematical dials to fix it. lr=0.0005 (Learning Rate) tells Adam to make very tiny, careful adjustments.

    epochs = 20
    
    print("Starting training phase...")
    net.train() 


    """new for structure knowledgeconveyor_belt = [
    ("Alice", "Bob", 5.0),
    ("Charlie", "David", 2.5),
    ("Eve", "Frank", 10.0)
]
    for package in conveyor_belt:
    print(package)
    
    # Output:
    # ('Alice', 'Bob', 5.0)
    # ('Charlie', 'David', 2.5)

    for sender, receiver, weight in conveyor_belt:
        print(f"{sender} is sending {weight}kg to {receiver}")

    # Output:
    # Alice is sending 5.0kg to Bob
    # Charlie is sending 2.5kg to David

    enumerate() is a built-in Python function that takes a collection of items (like a list or a dataloader) and attaches an automatic counter to each item as you loop through them.

    for i, package in enumerate(conveyor_belt):
        print(f"Package number {i} is {package}")

    # Output:
    # Package number 0 is ('Alice', 'Bob', 5.0)
    # Package number 1 is ('Charlie', 'David', 2.5)

    for i, (sender, receiver, weight) in enumerate(conveyor_belt):
        print(f"Batch {i}: {sender} -> {receiver}")

    # Output:
    # Batch 0: Alice -> Bob
    # Batch 1: Charlie -> David
    # Batch 2: Eve -> Frank
    """


    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (img0, img1, label) in enumerate(train_dataloader, 0):#0 is by default index
            img0, img1, label = img0.to(device), img1.to(device), label.to(device)

            """The Whiteboard Analogy
            Imagine a student solving a complex math problem on a whiteboard.

            The student looks at a problem (a batch of data).

            They do the math and write the answer on the board (the gradients).

            The teacher checks it and adjusts the student's grade (the optimizer updates the weights).

            If the student moves on to the second math problem without erasing the whiteboard first, they will end up writing the new numbers right on top of the old ones. The math becomes a chaotic, unreadable mess, and they will get the wrong answer.

            optimizer.zero_grad() is the eraser. It wipes the mathematical whiteboard completely clean so the model can calculate fresh, accurate directions for the new batch of images."""

            optimizer.zero_grad()

            """In Python, if you put a magic method named __call__ inside a class, it gives the resulting object the special ability to be triggered exactly like a function.

            Because your SiameseParkinsonNetwork inherited from PyTorch's nn.Module, it silently inherited PyTorch's highly complex __call__ function.

            2. The Secret Handshake (The forward function)
            When you write net(img0, img1), here is the exact chain of events that happens in the fraction of a millisecond:

            Python sees the parentheses and triggers the hidden __call__ function inherited from PyTorch.

            PyTorch briefly takes control. It sets up the invisible mathematical tracking system (the computational graph) so it can calculate the calculus gradients later.

            Once the setup is done, PyTorch automatically looks inside your code for a function named forward and passes img0 and img1 directly into it.

            Your forward function does the actual work (passing the images through the convolution layers), and hands the results (output1, output2) back out."""
            
            output1, output2 = net(img0, img1)
            
            final_loss_contrastive = criterion(output1, output2, label)
            final_loss_contrastive.backward()
            optimizer.step()

            """loss_contrastive isn't just a number? It is a PyTorch Tensor with a massive, invisible "breadcrumb trail" (computational graph) attached to it so .backward() can do its calculus.

            That breadcrumb trail takes up a huge amount of memory (RAM/VRAM) on your computer.

            If you just wrote epoch_loss += loss_contrastive, you wouldn't just be throwing the number 2.45 into the bucket. You would be throwing the number plus the massive breadcrumb trail. If you do this for 1,000 batches, you will stack up 1,000 massive breadcrumb trails in your bucket.

            Your computer's memory will fill up completely, and your program will crash with an "Out of Memory" (OOM) error.

            3. The "Safety Scissors" (.item())
            The .item() command acts like a pair of safety scissors.

            When you call loss_contrastive.item(), you are telling PyTorch:

            "I only want the raw, standard Python number (like 2.45). Cut off the massive PyTorch breadcrumb trail and leave it behind."

            By using .item(), you safely drop a lightweight, normal number into your epoch_loss bucket. The heavy breadcrumb trail gets deleted by the computer, keeping your memory completely clean for the next batch of images."""
            
            epoch_loss += loss_contrastive.item()
            
        print(f"Epoch [{epoch+1}/{epochs}] | Average Loss: {epoch_loss / len(train_dataloader):.4f}")

    os.makedirs(weights_dir, exist_ok=True)
    
    # Dynamically names the output file based on your input!
    save_path = os.path.join(weights_dir, f"siamese_{test_type}.pth") 
    torch.save(net.state_dict(), save_path)
    print(f"Success! Weights saved to: {save_path}\n")

if __name__ == "__main__":
    #This line tells Python: "Only run the code below this line IF I am running this specific file directly. If another file is just importing me to borrow my functions, ignore this section completely."
    # This captures what you type in the terminal
    parser = argparse.ArgumentParser(description="Train Parkinson's Siamese Network")


    """strict rules about what the user is allowed to type.

    '--type': The user must use this exact flag.

    type=str: Whatever comes after the flag must be treated as text (a string).

    required=True: The script will refuse to run if the user forgets this flag.

    help: If the user types python script.py --help, this exact message will pop up to guide them."""
    parser.add_argument(
        '--type', 
        type=str, 
        required=True, 
        help="The name of the subfolder to train (e.g., spiral, meander, circle)"
    )
    args = parser.parse_args()
    #What it does: This line actually executes the rules. It reads the terminal, checks that you followed the rules (meaning you included --type), grabs the word you typed, and saves it inside the args box.
    
    # Pass the terminal argument into the training function
    train_model(args.type)