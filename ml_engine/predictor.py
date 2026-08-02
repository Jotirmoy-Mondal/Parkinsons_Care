import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

# Import the exact architecture you provided
from .architecture import SiameseParkinsonNetwork

# Standard transform matching your training configuration
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def clean_and_threshold_image(image_path):
    """
    Applies OpenCV adaptive thresholding to convert mobile canvas uploads
    or paper photos into clean, normalized binary line art.
    """
    img = cv2.imread(image_path, cv2.COLOR_BGR2GRAY)
    if img is None:
        raise FileNotFoundError(f"Could not read image at path: {image_path}")
        
    # Adaptive thresholding to force white background and crisp black lines
    binary_img = cv2.adaptiveThreshold(
        img, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    # Smooth resize to the exact 224x224 shape expected by HandwritingFeatureExtractor
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
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Missing weight metrics for '{test_type}' at {weights_path}. Run training first.")
        
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # 2. Grab the reference healthy baseline template from your raw dataset folder
    base_dir = os.path.dirname(__file__)
    anchor_dir = os.path.join(base_dir, 'data', 'raw', test_type, 'healthy')
    
    if not os.path.exists(anchor_dir):
        raise FileNotFoundError(f"Reference baseline template directory not found at: {anchor_dir}")
        
    valid_anchors = [f for f in os.listdir(anchor_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not valid_anchors:
        raise FileNotFoundError(f"No baseline image templates found inside: {anchor_dir}")
        
    anchor_full_path = os.path.join(anchor_dir, valid_anchors[0])

    # 3. Normalize and process both images through the custom OpenCV threshold pipeline
    img_user = clean_and_threshold_image(uploaded_image_path)
    img_anchor = clean_and_threshold_image(anchor_full_path)
    
    # Convert clean PIL frames into analytical tensors and add a batch dimension
    tensor_user = transform(img_user).unsqueeze(0).to(device)
    tensor_anchor = transform(img_anchor).unsqueeze(0).to(device)

    # 4. Extract 128-dimensional coordinate embeddings and measure spatial distance
    with torch.no_grad():
        output_user, output_anchor = model(tensor_user, tensor_anchor)
        distance = F.pairwise_distance(output_user, output_anchor).item()

    # 5. Convert Euclidean distance into a human-readable Stability Percentage
    # Perfect match yields 100%. Highly unstable layouts fall toward 0%.
    stability_score = max(0.0, min(100.0, (1.0 - distance) * 100))
    return round(stability_score, 2)