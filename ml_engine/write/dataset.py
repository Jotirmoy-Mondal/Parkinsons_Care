"""
=============================================================================
LOGIC FLOWCHART: ParkinsonSiameseDataset
=============================================================================
[ INIT PHASE (Preparing the Dataset) ]
Input: Directory Paths (img_dir, subfolder)
     |
     v
Scan 'healthy' (1) & 'patient' (2) subfolders
     |--> Read filenames, extract ID_PATIENT & CLASS_TYPE
     |--> Store FULL_PATH and metadata in self.data list
     |--> Separate indices into healthy_indices & parkinson_indices
     |--> Extract unique patients into healthy_patients & parkinson_patients
     v
Dataset Ready (Length = Total Images)

[ GET ITEM PHASE (Fetching Pairs during Training) ]
Input: idx (Integer)
     |
1. Fetch Anchor Image metadata (Image A) at idx
     |
2. Flip a coin (is_same_class? True/False)
     |--> [True: POSITIVE PAIR]
     |      |--> Pick random image from SAME class pool
     |      |--> Ensure different ID_PATIENT (if >1 patient available)
     |      |--> Label = 0.0 (Meaning "Distance is 0 / Similar")
     |
     |--> [False: NEGATIVE PAIR]
     |      |--> Pick random image from DIFFERENT class pool
     |      |--> Label = 1.0 (Meaning "Distance is 1 / Dissimilar")
     |
3. Load both images from disk
     |
4. Convert to Grayscale ('L') & Apply transforms (Resize, Tensor, Normalize)
     |
Output: img_a (Tensor), img_b (Tensor), label (Tensor)
=============================================================================
"""

import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ParkinsonSiameseDataset(Dataset):
    def __init__(self, img_dir, subfolder="", transform=None):
        # os.path.join():
        # TAKES: Two or more strings representing folder/file names.
        # RETURNS: A single concatenated string with correct OS-specific slashes (e.g., 'data/raw\spiral').
        self.base_dir = os.path.join(img_dir, subfolder)
        
        # Define the paths to the sub-subfolders you created
        self.folders_to_scan = {
            1: os.path.join(self.base_dir, 'healthy'),
            2: os.path.join(self.base_dir, 'patient')
        }
        
        self.data = []
        self.healthy_indices = []
        self.parkinson_indices = []
        
        # Scan both directories
        idx = 0
        
        # dict.items():
        # TAKES: Nothing (called on a dictionary).
        # RETURNS: A view object that yields (key, value) tuple pairs for iteration.
        for class_type, folder_path in self.folders_to_scan.items():
            
            # os.path.exists():
            # TAKES: A string representing a file or directory path.
            # RETURNS: A boolean (True if the path exists on the hard drive, False otherwise).
            if not os.path.exists(folder_path):
                
                # print():
                # TAKES: One or more objects (usually strings).
                # RETURNS: None (it outputs the text to the console).
                print(f"Warning: Folder not found at {folder_path}. Skip check if not using it yet.")
                continue
                
            # os.listdir():
            # TAKES: A string representing a directory path.
            # RETURNS: A list of strings containing the names of the files in that directory.
            
            # str.endswith():
            # TAKES: A string or a tuple of strings (like ('.jpg', '.png')).
            # RETURNS: A boolean (True if the string ends with any of the specified suffixes).
            all_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
            
            for filename in all_files:
                
                # str.split():
                # TAKES: A separator string (like '-').
                # RETURNS: A list of strings broken apart at the separator.
                parts = filename.split('-') # split right and left word of "-" and return in a list
                
                # len():
                # TAKES: A sequence or collection (like a list, string, or dictionary).
                # RETURNS: An integer representing the number of items in that sequence.
                if len(parts) < 2:#[spiral, H1.png]
                    continue 
                    
                patient_part = parts[-1].split('.')[0] # e.g., "H1" or "P4"
                
                # Store the FULL path relative to the root so PIL can open it later
                full_image_path = os.path.join(folder_path, filename)
                
                # list.append():
                # TAKES: An object to add to the list.
                # RETURNS: None (it modifies the original list in place).
                self.data.append({
                    'FULL_PATH': full_image_path,
                    'CLASS_TYPE': class_type, # store key value 1 or 2
                    'ID_PATIENT': patient_part
                })
                
                if class_type == 1:
                    self.healthy_indices.append(idx)
                else:
                    self.parkinson_indices.append(idx)
                idx += 1

        # NOTE: Added the precomputation of unique patients from our previous discussion 
        # so `pool_patients` in __getitem__ works correctly!
        
        # set():
        # TAKES: An iterable (like a list or a generator expression).
        # RETURNS: A new set object containing only the UNIQUE elements from the iterable.
        self.healthy_patients = set(self.data[i]['ID_PATIENT'] for i in self.healthy_indices)
        self.parkinson_patients = set(self.data[i]['ID_PATIENT'] for i in self.parkinson_indices)

        print(f"Successfully loaded {len(self.data)} images from nested structure.")
        print(f"-> Healthy Subfolder count: {len(self.healthy_indices)}")
        print(f"-> Patient Subfolder count: {len(self.parkinson_indices)}")

        self.transform = transform if transform else transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        """
        Fetches an image pair (Anchor and a randomly chosen Positive/Negative) 
        and their distance label for Siamese Network training.
        """
        # 1. Get the Anchor image (Image A) metadata
        anchor_info = self.data[idx]
        anchor_path = anchor_info['FULL_PATH']
        anchor_class = anchor_info['CLASS_TYPE']
        anchor_patient = anchor_info['ID_PATIENT']
        
        # random.choice():
        # TAKES: A non-empty sequence (like a list or tuple).
        # RETURNS: A randomly selected single element from that sequence.
        # 2. Randomly decide whether to pair with the same class or a different class
        is_same_class = random.choice([True, False])
        
        if is_same_class:
            # --- POSITIVE PAIR (Same Class) ---
            pool = self.healthy_indices if anchor_class == 1 else self.parkinson_indices
            pool_patients = self.healthy_patients if anchor_class == 1 else self.parkinson_patients
            
            # SAFEGUARD: If only 1 unique patient exists in this class, we MUST use them
            if len(pool_patients) <= 1:
                paired_idx = random.choice(pool)
            else:
                # We have at least 2 patients, so we can safely mandate a different patient
                while True:
                    paired_idx = random.choice(pool)
                    if self.data[paired_idx]['ID_PATIENT'] != anchor_patient:
                        break
                        
            label = 0.0  # 0 distance for similar images
            
        else:
            # --- NEGATIVE PAIR (Different Class) ---
            # No infinite loop risk here because the Anchor and Paired patient will always 
            # have different CLASS_TYPEs, inherently making them different patients.
            pool = self.parkinson_indices if anchor_class == 1 else self.healthy_indices
            paired_idx = random.choice(pool)
            label = 1.0  # High distance for dissimilar images
            
        paired_path = self.data[paired_idx]['FULL_PATH']
    
        # Image.open():
        # TAKES: A string path to an image file.
        # RETURNS: A PIL Image object (lazily loaded from the hard drive).
        
        # 3. Load, convert to Grayscale ('L'), and apply transformations
        with Image.open(anchor_path) as img_a_raw, Image.open(paired_path) as img_b_raw:
            
            # Image.convert():
            # TAKES: A string indicating the desired color mode ('L' means 8-bit pixels, black and white).
            # RETURNS: A new PIL Image object converted to the requested color mode.
            img_a = img_a_raw.convert('L')
            img_b = img_b_raw.convert('L')
            
            if self.transform is not None:
                img_a = self.transform(img_a)
                img_b = self.transform(img_b)
        
        # torch.tensor():
        # TAKES: Data (like a list or number) and optionally a specific data type (dtype).
        # RETURNS: A PyTorch tensor containing the data, ready for neural network calculations.
        return img_a, img_b, torch.tensor([label], dtype=torch.float32)
