"""1.Initialize AI Framework:The script checks your hardware (GPU vs CPU), builds the empty SiameseParkinsonNetwork structure, and loads your saved .pth weights. It immediately calls .eval() to lock the brain and prevent accidental learning.
2.Retrieve the Healthy Baseline (Anchor):The script navigates to your data/raw/ folders, finds the specific test category (e.g., "spiral"), and selects a verified healthy drawing to act as the "answer key."
3.OpenCV Image Cleaning:Both the user's uploaded image and the healthy anchor image are sent through clean_and_threshold_image(). They are converted to grayscale, stripped of shadows using adaptive thresholding, and rigidly resized to 224x224 pixels.4.PyTorch Tensor Conversion:The clean images are passed through your transforms. They are converted into PyTorch Tensors, mathematically normalized, and given a "fake" batch dimension using .unsqueeze(0) so the AI thinks it's receiving a batch of size 1.
5.Feature Extraction (Forward Pass):The script turns off the calculus engine using torch.no_grad() to save memory. It passes both image tensors through the AI network to extract their final 128-dimensional mathematical coordinates.
6.Calculate Euclidean Distance:Using F.pairwise_distance, the script measures the exact straight-line mathematical distance between the user's output coordinates and the healthy anchor's output coordinates.
7.Calculate Final Score:The raw distance metric is inverted and mathematically clamped between 0.0 and 100.0 to generate the final human-readable Stability Score, which is rounded to two decimal places and returned to the user."""


import os
import cv2
#Computer Vision: It is the field of Artificial Intelligence that focuses on teaching computers how to "see" and understand digital images and videos (like removing shadows, finding edges, or recognizing faces).
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
"""What Image does in your code

The Bridge (PIL): Scoops up OpenCV's raw code and packages it into a standard image object so the rest of Python can read it.

In your script, Image acts as the translator between two completely different software systems (OpenCV and PyTorch).

Here is exactly why you needed it:

The OpenCV format: When you used cv2 to clean the image and remove the shadows, OpenCV stored the result as a raw block of numbers called a NumPy Array.

The PyTorch requirement: Your PyTorch image transformer (transforms.ToTensor()) prefers to receive a standard Python image object, not a raw block of numbers.

If you tried to hand OpenCV's raw numbers directly to PyTorch, it could cause formatting issues.

The Bridge
At the very end of your clean_and_threshold_image function, you wrote this line:

Python
return Image.fromarray(resized_img)
This uses the PIL Image tool to scoop up OpenCV's raw number array (resized_img) and instantly convert it back into a standard, formatted image object. Now, when you pass it down to PyTorch in the next step, PyTorch accepts it perfectly."""
import torchvision.transforms as transforms
#The Mathematician (torchvision): Takes that image, turns it into a PyTorch Tensor, and squishes all the colors into the perfect mathematical range (-1.0 to 1.0).
# Import the exact architecture you provided
from architecture import SiameseParkinsonNetwork

# Standard transform matching your training configuration
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
#here compose pass the output of transforms.ToTensor() to transforms.Normalize(mean=[0.5], std=[0.5])

def clean_and_threshold_image(image_path):
    """
    Applies OpenCV adaptive thresholding to convert mobile canvas uploads
    or paper photos into clean, normalized binary line art.
    """
    img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)

    """The imread function takes two arguments:

filename (Required): A text string of the exact file path where your image is saved. (e.g., "user_upload.jpg").

flag (Optional): An integer or "passcode" that tells OpenCV how you want the colors handled.

If you leave this blank, it defaults to 1 (which means cv2.IMREAD_COLOR - read it in full color).

If you pass 0 (which is cv2.IMREAD_GRAYSCALE), it strips the color away immediately.

What it RETURNS (The Outputs)
When the function finishes running, it returns one of two possible things:

1. A NumPy Array (Success)
If it successfully finds and opens the photo, it returns a NumPy Array. This is a massive, multi-dimensional grid of numbers representing every single pixel in the image.

2. None (Failure)
If you give it a bad file path, or if the image is corrupted, it does not crash your program right away. Instead, it quietly returns the Python value None (meaning "nothing")."""

    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")
        
    # Adaptive thresholding to force white background and crisp black lines

    """What it TAKES (The Inputs)
It requires exactly 6 specific arguments, in this exact order:

src (The Image): The input image. Crucial rule: It must already be a 1-channel Grayscale image. If you pass a full-color image here, the function will crash.

maxValue: The color value you want to assign to the pixels that "pass" the test. In 99% of cases, this is 255 (pure white).

adaptiveMethod: The mathematical formula used to calculate the lighting in a small area.

cv2.ADAPTIVE_THRESH_MEAN_C: Takes a simple average of the nearby pixels.

cv2.ADAPTIVE_THRESH_GAUSSIAN_C: Uses a bell-curve average, giving more importance to the center pixels (usually yields cleaner results).

thresholdType: How to color the result.

cv2.THRESH_BINARY: Background becomes white, pen strokes become black.

cv2.THRESH_BINARY_INV: Background becomes black, pen strokes become white.

blockSize: The size of the "magnifying glass" grid used to check local lighting (e.g., 11 means an 11x11 pixel square). It must be an odd number greater than 1 (3, 5, 7, 11, etc.).

C (The Constant): A simple number (like 2) subtracted from the final math to fine-tune the contrast and reduce static/noise.

What it RETURNS (The Output)
It returns exactly one NumPy Array.

This new array is the exact same width and height as your original image, but it has been permanently converted into a Binary Image.

"Binary" means there are no longer hundreds of shades of gray. Every single pixel in this new array has been forced to be exactly one of two numbers:

0 (Pure Black)

255 (Pure White)

There is no in-between. It completely erases the shadows and leaves you with a perfectly crisp, high-contrast map of your patient's drawing."""
    binary_img = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Smooth resize to the exact 224x224 shape expected by HandwritingFeatureExtractor
    """The cv2.resize function acts as the shape-shifter for your image. In your code, it takes exactly three things to do its job:

    What it TAKES (The Inputs)
    binary_img (The Source):
    This is the pure black-and-white image you just created in the previous step using the threshold tool.

    (224, 224) (The Target Size):
    This is a strict requirement for PyTorch. It tells OpenCV to force the image to be exactly 224 pixels wide and 224 pixels tall. It does not matter if the patient uploaded a massive 4K photo or a tiny square; this forces it into the exact shape the AI’s input layer expects.

    interpolation=cv2.INTER_AREA (The Resizing Math):
    When you shrink a massive photo down to a tiny 224x224 square, you have to delete thousands of pixels. "Interpolation" is the specific math formula the computer uses to decide which pixels to keep and which to throw away.

    cv2.INTER_AREA is a highly specific formula designed only for shrinking images. Instead of just deleting pixels randomly, it averages them together. This ensures that the thin, delicate pen strokes of the Parkinson's drawing don't accidentally get erased or broken when the image gets smaller.

    The Return Step
    After cv2.resize finishes its math, it spits out a new 224x224 NumPy array.

    Your final line, return Image.fromarray(resized_img), catches that array, packages it into a standard Python Image object (using the PIL library we discussed earlier), and hands it back to the main script so it can be sent to the AI."""
    resized_img = cv2.resize(binary_img, (224, 224), interpolation=cv2.INTER_AREA)
    return Image.fromarray(resized_img)

def evaluate_stability(uploaded_image_path, test_type):
    """
    Compares a user upload against a baseline healthy template image.
    Returns a calculated stability score out of 100.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Instantiate the twin framework structure and load specific trained weights
    model = SiameseParkinsonNetwork().to(device)
    weights_path = os.path.join(os.path.dirname(__file__), 'weights', f'siamese_{test_type}.pth')
    #1. The current folder: os.path.dirname(__file__)

    #2. The sub-folder: 'weights'

    #3. The file name: f'siamese_{test_type}.pth'
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weight metrics for '{test_type}' at {weights_path}. Run training first.")



    """This line of code performs a **"Brain Transplant"** on your AI.

    ### 1. `torch.load()` (The File Reader)

    This function is PyTorch's official file reader. It goes to your hard drive, opens the saved `.pth` file, and extracts the data.

    **What it TAKES:**

    * **`weights_path`:** The exact file path on your computer where the `.pth` file is located.
    * **`map_location=device`:** This is a brilliant safety feature. If you trained your AI on a massive cloud GPU, PyTorch saved the file as a "GPU file." If you try to open that file on a basic laptop that only has a CPU, PyTorch will crash. `map_location=device` tells PyTorch: *"I don't care what hardware this was trained on. Translate the data to perfectly match whatever hardware (`device`) I am currently using."*

    **What it RETURNS:**
    It returns a **State Dictionary** (`state_dict`). This is simply a massive, organized Python list containing millions of highly specific numbers (the "weights" and "biases") that the AI calculated during its 20 epochs of training.

    ### 2. `model.load_state_dict()` (The Surgeon)

    **What it TAKES:**

    * It takes the **State Dictionary** (the list of learned numbers) provided by `torch.load()`.

    **What it DOES:**
    It acts like a surgeon, taking every single number from that saved list and perfectly injecting it into the correct layers of your empty `model`. It maps the saved memory perfectly to the architecture.

    ---

    ### The Final Step: `model.eval()`

    In PyTorch, a model can be in one of two modes:

    1. **Training Mode (`model.train()`):** The model is actively updating its weights, changing its math, and "learning."
    2. **Evaluation Mode (`model.eval()`):** The model's brain is completely locked.

    By calling `model.eval()`, you are telling the AI: *"Your training is completely finished. Do not try to learn from this new patient's drawing. Just use your locked, saved memories to grade it."*"""
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Grab the reference healthy baseline template from your raw dataset folder
    base_dir = os.path.dirname(__file__)
    anchor_dir = os.path.join(base_dir, 'data', 'raw', test_type, 'healthy')
    
    if not os.path.exists(anchor_dir):
        raise FileNotFoundError(f"Reference baseline template directory not found at: {anchor_dir}")
        


    """The Everyday Example
    basket = ["apple", "banana", "cherry"]
    new_box = []                    # 1. Create an empty box

    for fruit in basket:            # 2. Look at every item in the basket
        new_box.append(fruit)       # 3. Put it in the new box

    The Short Way (List Comprehension):
    Python lets you condense those 3 steps into a single line using brackets [ ].

    Python
    basket = ["apple", "banana", "cherry"]

    new_box = [fruit for fruit in basket]
    Breaking Down Your Code
    When you wrote [f for f in os.listdir(anchor_dir)], you used this exact same shortcut. Here is how the syntax reads from left to right:

    [ ]: The outer brackets tell Python, "I am building a brand new list."

    f (The First One): This is what you want to put into the new list.

    for f in: This is the loop. It means, "look at every single item inside..."

    os.listdir(anchor_dir): This is the source box you are pulling from (in your case, the folder on your computer)."""


    #here f =file
    valid_anchors = [f for f in os.listdir(anchor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not valid_anchors:
        raise FileNotFoundError(f"No baseline image templates found inside: {anchor_dir}")
        
    #here 0 means first healthy image    
    anchor_full_path = os.path.join(anchor_dir, valid_anchors[0])

    # 3. Normalize and process both images through the custom OpenCV threshold pipeline
    """What it RETURNS (The Output)
    It returns a PIL.Image object (a standard Python image).

    When the function finishes running, it hands back an image that has been completely transformed. Because of the code you wrote inside that function, the returned image is guaranteed to be:

    Grayscale (All color stripped away).

    High-Contrast Binary (Shadows removed, leaving only pure black background and pure white pen strokes).

    Perfectly Sized (Squished or stretched into a strict 224x224 pixel square).

    Translated (Converted from OpenCV's raw math array back into a standard Python image using Image.fromarray)."""
    img_user = clean_and_threshold_image(uploaded_image_path)
    img_anchor = clean_and_threshold_image(anchor_full_path)
    
    # Convert clean PIL frames into analytical tensors and add a batch dimension

    """1. transform(img_user)
    This is the assembly line I built earlier using torchvision.transforms.Compose.

    What it TAKES: The clean PIL.Image 
    i just generated in the previous step.

    What it RETURNS: A 3D PyTorch Tensor.

    What it DOES: It translates the Python image into a mathematical matrix, squishes the colors between -1.0 and 1.0, and outputs a 3D shape of [1, 224, 224] (1 color channel, 224 width, 224 height).

    2. .unsqueeze(0)
    This is the "fake box" creator.

    What it TAKES: The number 0. This simply tells PyTorch where to add the fake dimension (index 0 means at the very front).

    What it RETURNS: A 4D PyTorch Tensor.

    What it DOES: Remember the factory analogy? The AI refuses to process a single loose image. It only accepts "boxes" (batches) of images. This wraps a fake batch dimension around your tensor, changing its shape from [1, 224, 224] to [1, 1, 224, 224] so the neural network won't crash."""


    tensor_user = transform(img_user).unsqueeze(0).to(device)
    tensor_anchor = transform(img_anchor).unsqueeze(0).to(device)

    # 4. Extract 128-dimensional coordinate embeddings and measure spatial distance

    """from architecture.py
    def forward(self, image_yesterday, image_today):
        # Pass Image A through the network to get its coordinates
        embedding_a = self.twin_cnn(image_yesterday)
        
        # Pass Image B through the SAME network to get its coordinates
        embedding_b = self.twin_cnn(image_today)
        
        # Return both sets of coordinates so the Loss Function can compare them
        return embedding_a, embedding_b
        
        The .item() command is a built-in PyTorch extractor. Its only job is to rip a standard Python number out of a PyTorch Tensor wrapper.
        If you printed it without .item(), it would look like this:

    Python
    # What PyTorch outputs
    tensor([3.1415], device='cuda:0')
        """

    with torch.no_grad():
        output_user, output_anchor = model(tensor_user, tensor_anchor)
        distance = F.pairwise_distance(output_user, output_anchor).item()


    # 5. Convert Euclidean distance into a human-readable Stability Percentage
    # Perfect match yields 100%. Highly unstable layouts fall toward 0%.
    # 5. Convert Euclidean distance into a progressive exponential decay score
    # k controls the steepness of the drop. 1.5 ensures smooth tracking for larger distances.
    k=1.5
    stability_score = 100.0*math.exp(-k*distance)
    return round(stability_score, 2)