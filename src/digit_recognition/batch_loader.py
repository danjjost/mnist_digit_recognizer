from decimal import Decimal
import os
from random import Random
from typing import Optional
from config import Config

import numpy as np
from PIL import Image

from src.digit_recognition.MNISTImage import MNISTImage
        
class BatchLoader():
    def __init__(self, config: Optional[Config], random: Optional[Random]):
        self.config = config or Config()
        self.random = random or Random()
    
    def get_training_batch(self, batch_size: int) -> list[MNISTImage]:
        print(f'Loading training batch of size {batch_size}.')
        path = self.config.mnist_training_folder
        
        return self.get_batch_from_path(path, batch_size)
    
    def get_testing_batch(self, batch_size: int) -> list[MNISTImage]:
        print(f'Loading testing batch of size {batch_size}.')
        path = self.config.mnist_testing_folder
        
        return self.get_batch_from_path(path, batch_size)
        
    def get_batch_from_path(self, path: str, batch_size: int) -> list[MNISTImage]:
        mnist_images: list[MNISTImage] = []
        
        for _ in range(batch_size):
            image = self.get_random_image(path)
            mnist_images.append(image)
            
        return mnist_images            
    
    def get_random_image(self, base_path: str) -> MNISTImage:
        random_digit = self.random.randint(0, 9)
        path = base_path + str(random_digit)
        
        all_files_in_folder = os.listdir(path)
        random_file = self.random.choice(all_files_in_folder)
        
        file_path = path + "/" + random_file
        
        image_array = self.load_image(file_path)
        
        return MNISTImage(image_array, random_digit)
    
    def load_image(self, file_path: str) -> list[Decimal]:
        img = Image.open(file_path) # type: ignore

        img_gray = img.convert('L')
        img_array = np.array(img_gray)
        img_array_float = img_array.astype(np.float32) / 255.0
        
        precision = Config().decimal_precision
        
        return [Decimal(x.item()).quantize(Decimal(10) ** -precision) for x in img_array_float.flatten()]