# scripts/dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

import sys
sys.path.append('e:/version_2/vit_detector')
import config

class ViTDetDataset(Dataset):
    def __init__(self, root_dir, split='train', transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images and labels.
                               Assumes a structure like:
                               - root_dir/images/split/
                               - root_dir/labels/split/
            split (string): 'train' or 'val'.
            transforms (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        self.split = split

        self.img_dir = os.path.join(root_dir, 'images', split)
        self.label_dir = os.path.join(root_dir, 'labels', split)

        self.image_files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.jpg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Use PIL to get image dimensions for normalization
        image = Image.open(img_path).convert("RGB")
        
        # Corresponding label file
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.label_dir, label_name)

        boxes = []
        labels = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    
                    class_id = int(parts[0])
                    
                    # Check if this is standard YOLO format (5 parts) or polygon format (more parts)
                    if len(parts) == 5:
                        # Standard YOLO format: [class, x_center, y_center, width, height] (normalized)
                        cx, cy, w, h = map(float, parts[1:5])
                    elif len(parts) > 5:
                        # Polygon format: convert to bounding box
                        # Parts are: class_id, x1, y1, x2, y2, x3, y3, x4, y4, ... (polygon points)
                        coords = list(map(float, parts[1:]))
                        
                        # Extract x and y coordinates
                        x_coords = coords[::2]  # Every even index
                        y_coords = coords[1::2]  # Every odd index
                        
                        # Get bounding box from polygon
                        x_min, x_max = min(x_coords), max(x_coords)
                        y_min, y_max = min(y_coords), max(y_coords)
                        
                        # Convert to center format
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min
                    else:
                        continue  # Skip invalid lines
                    
                    # Ensure coordinates are in valid range [0, 1]
                    cx = max(0, min(1, cx))
                    cy = max(0, min(1, cy))
                    w = max(0, min(1, w))
                    h = max(0, min(1, h))
                    
                    # Only add if box has valid dimensions
                    if w > 0 and h > 0:
                        boxes.append([cx, cy, w, h])
                        labels.append(class_id)

        target = {}
        target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
        target["labels"] = torch.as_tensor(labels, dtype=torch.int64)

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target

# Example of a transform function (to be expanded in utils.py or train.py)
class SimpleTransform:
    def __init__(self, normalize=True):
        self.normalize = normalize

    def __call__(self, image, target):
        # To tensor
        image = F.to_tensor(image)
        
        # Normalize
        if self.normalize:
            image = F.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        return image, target


if __name__ == '__main__':
    # --- Test the Dataset ---
    print("Testing ViTDetDataset...")
    
    # Create a dummy transform
    transforms = SimpleTransform()

    # Create dataset instance
    # Make sure the path in config.DATA_PATH is correct
    try:
        dataset = ViTDetDataset(root_dir=config.DATA_PATH, split='train', transforms=transforms)
        print(f"Successfully loaded dataset. Found {len(dataset)} samples.")
        
        # Get one sample
        if len(dataset) > 0:
            image, target = dataset[0]
            print("\nSample 0:")
            print(f"  - Image shape: {image.shape}")
            print(f"  - Target boxes:\n{target['boxes']}")
            print(f"  - Target labels:\n{target['labels']}")
            print(f"  - Box format is (cx, cy, w, h) normalized.")
        else:
            print("Dataset is empty. Check your data paths and split name.")

    except FileNotFoundError:
        print("\nError: Dataset directory not found.")
        print(f"Please ensure the path '{config.DATA_PATH}' is correct and contains 'images/train' and 'labels/train' subdirectories.")
