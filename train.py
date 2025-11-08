import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import json
from data.dataset import DiffusionDataset 
import os
from DiffusionModule import Diffusion
from model.clip_import import CLIPImageEncoder, CLIPTextEncoder, CLIPLoss 
import math
from torch.cuda.amp import autocast, GradScaler



def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

class ExponentialMovingAverage_EMA:
    """Maintains moving average of model parameters for stable validation"""
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
                
                
                
                
                
def train_model(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    dataset = DiffusionDataset(image_dir=config['image_dir'], captions_file=config['captions_file'], image_size=config['image_size'])  
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    
    
    model = Diffusion(
        timesteps=config['timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'] ).to(device)
    
    model_name = config['image&text_encoding_model'] # clip 
    
    text_encoder = CLIPTextEncoder(model_name=model_name, device=device)
    image_encoder = CLIPImageEncoder(model_name=model_name, device=device)
    clip_loss = CLIPLoss(model_name=model_name, device=device)
    
    
    clip_loss_fn= clip_loss.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'] ,
                            weight_decay=config.get('weight_decay', 1e-2),
                            betas=(0.9, 0.999))


    total_iterations = config['total_iterations']
    warmup_iterations = config.get('warmup_iterations', 0.1 * total_iterations)
    scheduler=get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_iterations,
                                            num_training_steps=total_iterations)
    
    
    scaler = GradScaler(enabled=config.get('mixed_precision', True))

    
    max_grad_norm = config.get('max_grad_norm', 1.0)


    ema = ExponentialMovingAverage_EMA(model, decay=config.get('ema_decay', 0.9999))

    iteration = 0
    while iteration < total_iterations:
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            captions = batch['caption']

            batch_size = images.size(0)

            
            t = torch.randint(0, config['timesteps'], (batch_size,), device=device).long()

            
            z_0 = model.encode(images)

            
            inputs = text_encoder(text=captions, return_tensors="pt", padding=True, truncation=True).to(device)
            
            with torch.no_grad():
                text_embeddings = clip_model.get_text_features(**inputs)  # [batch_size, 768??]

            # Forward diffusion in latent space
            noise = torch.randn_like(z_0)
            z_t = model.q_sample_latent(z_0, t, noise=noise)

            # Predict noise with text conditioning
            predicted_noise = model.predict_latent_noise(z_t, t, context=text_embeddings)

            # Compute diffusion loss (MSE between predicted and actual noise)
            diffusion_loss = nn.functional.mse_loss(predicted_noise, noise)
            
            # Predict the clean latent (z_0_pred) from z_t using the reverse process
            z_0_pred, _ = model.p_sample_latent(z_t, t, clip_denoised=True)  # Get denoised latent
            
            predicted_images = model.decode(z_0_pred)  # Decode the denoised latent to image space
            # CLIP guidance: Encourage alignment between decoded images and captions
            with torch.no_grad():

                image_inputs = clip_processor(images= predicted_images, return_tensors="pt").to(device)
                image_embeddings = clip_model.get_image_features(**image_inputs)

            clip_loss = 1 - nn.functional.cosine_similarity(image_embeddings, text_embeddings).mean()
            total_loss = diffusion_loss + config['clip_loss_weight'] * clip_loss

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_loss += total_loss.item()

            if batch_idx % config['log_interval'] == 0:
                print(f"Epoch [{epoch}/{config['epochs']}], Batch [{batch_idx}/{len(dataloader)}], "
                    f"Diffusion Loss: {diffusion_loss.item():.4f}, CLIP Loss: {clip_loss.item():.4f}")

        # Save checkpoint
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch}/{config['epochs']}] completed. Avg Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(config['checkpoint_dir'], f"model_epoch_{epoch}.pth"))

    print("Training completed!")


def sample_images(model, config, num_samples=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        sampled_images = model.sample((num_samples, 3, 256, 256), device=device)  # Adjust shape as needed
    return sampled_images


    
    
