import torch
from torch.utils.data import Dataset
import json
import os
import random
from data.image_processor import ImageProcessor


class DiffusionDataset(Dataset):
    def __init__(self, image_dir, captions_file, image_size=512):
        
        self.image_dir = image_dir
        self.image_processor = ImageProcessor(image_size)
        
        
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
            
      
        self.samples = []
        print("Filtering dataset...")
        for item in self.captions:
            image_path = os.path.join(image_dir, item['image'])
            if os.path.exists(image_path):
                self.samples.append({
                    'image_path': image_path,
                    'caption': item['caption']
                })
        print(f"Found {len(self.samples)} total samples.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        
        image = self.image_processor.preprocess_image(sample['image_path'])
        
      
        while image is None:
            new_idx = random.randint(0, len(self) - 1)
            return self.__getitem__(new_idx)
    
  
        return {
            'image': image,
            'caption': sample['caption']   
        }