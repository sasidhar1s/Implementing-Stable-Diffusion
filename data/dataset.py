import torch
from torch.utils.data import Dataset
import json
import os
import random
from data.image_processor import ImageProcessor


class DiffusionDataset(Dataset):
    def __init__(self, image_dir, captions_file, image_size=256, caption_dropout_prob=0.1):
        self.image_dir = image_dir
        self.image_processor = ImageProcessor(image_size)
        self.caption_dropout_prob = caption_dropout_prob  
        
        with open(captions_file, 'r') as f:
            self.captions = json.load(f)
            
        self.samples = []
    
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
        
        
        if random.random() < self.caption_dropout_prob:
            caption = ""  # (guidance free training)
        else:
            caption = sample['caption']  # conditional training

        return {
            'image': image,
            'caption': caption  
        }
        
'''        
class ValidationDataset(Dataset):
    def __init__(self, prompts_file=None, max_prompts=None):
        
        
        with open(prompts_file, 'r') as f:
            data = json.load(f)
            if isinstance(data, list):
                prompts = data  # ["prompt1", "prompt2", ...]
            elif isinstance(data, dict) and 'prompts' in data:
                prompts = data['prompts']  # {"prompts": ["prompt1", ...]}
            else:
                prompts = [item['prompt'] for item in data]  # [{"prompt": "..."}, ...]
        
        
        
        if max_prompts is not None:
            prompts = prompts[:max_prompts]
        
        self.prompts = prompts
    
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return self.prompts[idx]
        
        
        
        '''