import torch
from transformers import CLIPTokenizer, CLIPTextModel, CLIPVisionModel, CLIPProcessor
import torch.nn as nn
import torch.nn.functional as F


class CLIPTextEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device='cuda'):
        
        self.device = device 
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.text_encoder = CLIPTextModel.from_pretrained(model_name)
        self.text_encoder.to(device)
        self.text_encoder.eval()
        
        
    def _encode(self, text_prompts, max_length=77):
        if isinstance(text_prompts, str):
            text_prompts = [text_prompts]
            
        tokens = self.tokenizer(
            text_prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        with torch.no_grad():
            outputs = self.text_encoder(
                
                input_ids=tokens.input_ids.to(self.device),
            
                attention_mask=tokens.attention_mask.to(self.device)
            )
            
            
        return outputs

    def encode_pooled(self, text_prompts, max_length=77):
        outputs = self._encode(text_prompts, max_length)
        return outputs.pooler_output

    def encode_last_hidden_state(self, text_prompts, max_length=77):
        outputs = self._encode(text_prompts, max_length)
        return outputs.last_hidden_state  
    
    def get_unconditional_embeddings(self, batch_size):
        
        return self.encode_last_hidden_state([""] * batch_size)  


class CLIPImageEncoder:
    def __init__(self, model_name="openai/clip-vit-large-patch14", device='cuda'): 
        self.device = device
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_encoder = CLIPVisionModel.from_pretrained(model_name)
        self.vision_encoder.to(device)
        self.vision_encoder.eval()

    def encode_images(self, images):
        
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.vision_encoder(**inputs)
        return outputs.pooler_output


class CLIPLoss(nn.Module):
    def __init__(self, model_name="openai/clip-vit-large-patch14", device='cuda'):  
        super().__init__()
        self.device = device
        self.text_encoder = CLIPTextEncoder(model_name, device)
        self.image_encoder = CLIPImageEncoder(model_name, device)
    
    def forward(self, images, text_prompts):
        image_features = self.image_encoder.encode_images(images)
        text_features = self.text_encoder.encode_pooled(text_prompts)  
        
        image_features = F.normalize(image_features, p=2, dim=-1)
        text_features = F.normalize(text_features, p=2, dim=-1)
        
        similarity = F.cosine_similarity(image_features, text_features)
        loss = 1.0 - similarity.mean()
        return loss
    
    