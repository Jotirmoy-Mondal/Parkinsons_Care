import os
import random
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class ParkinsonSiameseDataset(Dataset):
    def __init__(self, img_dir, subfolder="", transform=None):
        # Base directory (e.g., data/raw/spiral)
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
        for class_type, folder_path in self.folders_to_scan.items():
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found at {folder_path}. Skip check if not using it yet.")
                continue
                
            all_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
            
            for filename in all_files:
                parts = filename.split('-')
                if len(parts) < 2:
                    continue 
                    
                patient_part = parts[-1].split('.')[0] # e.g., "H1" or "P4"
                
                # Store the FULL path relative to the root so PIL can open it later
                full_image_path = os.path.join(folder_path, filename)
                
                self.data.append({
                    'FULL_PATH': full_image_path,
                    'CLASS_TYPE': class_type,
                    'ID_PATIENT': patient_part
                })
                
                if class_type == 1:
                    self.healthy_indices.append(idx)
                else:
                    self.parkinson_indices.append(idx)
                idx += 1

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

    def __getitem__(self, idx):
        img0_info = self.data[idx]
        img0_path = img0_info['FULL_PATH']
        img0_class = img0_info['CLASS_TYPE']
        img0_patient = img0_info['ID_PATIENT']
        
        should_get_same_class = random.randint(0, 1)
        
        if should_get_same_class:
            pool = self.healthy_indices if img0_class == 1 else self.parkinson_indices
            while True:
                img1_idx = random.choice(pool)
                img1_info = self.data[img1_idx]
                if img1_info['ID_PATIENT'] != img0_patient or len(pool) <= 1:
                    break
            label = 0.0  
        else:
            pool = self.parkinson_indices if img0_class == 1 else self.healthy_indices
            img1_idx = random.choice(pool)
            img1_info = self.data[img1_idx]
            label = 1.0  
            
        img1_path = img1_info['FULL_PATH']

        with Image.open(img0_path) as img0_raw, Image.open(img1_path) as img1_raw:
            img0 = self.transform(img0_raw.convert('L'))
            img1 = self.transform(img1_raw.convert('L'))
        
        return img0, img1, torch.tensor([label], dtype=torch.float32)