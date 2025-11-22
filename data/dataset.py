import torch
from torch.utils.data import Dataset
import json
import os
import random
from PIL import Image
from torchvision import transforms as T
from collections import defaultdict


class Crop_and_Resize:
    def __init__(self, image_size=256): 
        self.image_size = image_size
        
        
        self.transform = T.Compose([
            T.RandomCrop(image_size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True)
        ])

    def crop_resize(self, image_path):
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            
            return None
            
        
        if min(image.size) < self.image_size:
            
            return None

        return self.transform(image)


class DiffusionDataset(Dataset):
    def __init__(self, image_dir, captions_file, image_size=256, caption_dropout_prob=0.1):
        self.image_dir = image_dir
        self.image_processor = Crop_and_Resize(image_size)
        self.caption_dropout_prob = caption_dropout_prob
        
        with open(captions_file, 'r', encoding='utf-8') as f:
            captions_data = json.load(f)
            
        self.samples = []
    
        

        
        for img_name, caption_list in captions_data.items():
            
            
            image_path = os.path.join(image_dir, img_name)
            
    
            if os.path.exists(image_path):
            
                self.samples.append({
                    'image_path': image_path,
                    'captions': caption_list 
                })
        
        if not self.samples:
            raise ValueError("No valid image files found")
        
        
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
       
        while True:
         
            current_idx = idx % len(self.samples)
            sample = self.samples[current_idx]
            
            image = self.image_processor.crop_resize(sample['image_path'])
            
            if image is not None:
    
                break
            
            
            idx = random.randint(0, len(self) - 1)

        
        
        caption_list = sample['captions']
        selected_caption = random.choice(caption_list)
        
        
        if random.random() < self.caption_dropout_prob:
            final_caption = " "  
        else:
            final_caption = selected_caption

        return {
            'image': image,
            'caption': final_caption
        }