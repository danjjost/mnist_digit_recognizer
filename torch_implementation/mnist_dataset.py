import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import numpy as np


class MNISTDataset(Dataset):
    def __init__(self, image_path: str):
        """
        Initialize the MNISTDataset.
        
        Args:
            image_path (str): Path to the directory containing MNIST images. Should contain folders 0-9, each containing images of the corresponding digit.
        """
        self.image_paths = []
        self.labels = []

        # Collect image paths and corresponding labels
        for label in range(10):  # Assuming 10 classes for MNIST (0-9)
            label_dir = os.path.join(image_path, str(label))
            if os.path.exists(label_dir):
                for img_name in os.listdir(label_dir):
                    if img_name.lower().endswith('.jpg'):
                        img_path = os.path.join(label_dir, img_name)
                        self.image_paths.append(img_path)
                        self.labels.append(label)
    
    def __len__(self):
        """
        Return the total number of samples.
        """
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """
        Retrieve the image and label at the given index.
        
        Args:
            idx (int): Index of the sample to retrieve.
        
        Returns:
            torch.Tensor: The image tensor.
            int: The corresponding label.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('L')  # Convert to grayscale
        image = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        image = torch.tensor(image).unsqueeze(0)  # Add channel dimension
        
        return image, label
