import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import json
from data.dataset import DiffusionDataset 
import os
from DiffusionModule import Diffusion
from model.clip_import import  CLIPTextEncoder
import math
from torch.cuda.amp import autocast, GradScaler
from model.VAE_import import StableDiffusionVAE as VAE
from model.Unet import DenoisingModel 

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
    
    dataset = DiffusionDataset(image_dir=config['image_dir'], captions_file=config['captions_file'], 
                            image_size=config['image_size'],
                            caption_dropout_prob=config['caption_dropout_prob'])  
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=8, pin_memory=True)

    
    
    Diffusion_model = Diffusion(
        timesteps=config['timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'] ).to(device)
    
    
    Denoising_model = DenoisingModel(in_channels= config['latent_channels'],
                                    context_dim=config['context_dim']).to(device)
    
    
    # clip 
    
    text_encoder = CLIPTextEncoder(model_name=config['image&text_encoding_model'], device=device)
    #image_encoder = CLIPImageEncoder(model_name=config['image&text_encoding_model'], device=device)
    #clip_loss = CLIPLoss(model_name=config['image&text_encoding_model'], device=device)
    
    
    #clip_loss_fn= clip_loss.to(device)
    
    
    VAE_model = VAE(model_name=config['vae_model'], device=device)

    optimizer = optim.AdamW(Denoising_model.parameters(), lr=config['learning_rate'] ,
                            weight_decay=config.get('weight_decay', 1e-2),
                            betas=(0.9, 0.999))


    total_iterations = config['total_iterations']
    warmup_iterations = config.get('warmup_iterations', 0.1 * total_iterations)
    scheduler=get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_iterations,
                                            num_training_steps=total_iterations)
    
    
    scaler = GradScaler(enabled=config.get('mixed_precision', True))

    
    max_grad_norm = config.get('max_grad_norm', 1.0)


    ema = ExponentialMovingAverage_EMA(Denoising_model, decay=config.get('ema_decay', 0.9999))


    iteration = 0
    epoch = 0
    
    while iteration < total_iterations:
        DenoisingModel.train()
        

        for batch in (dataloader):
            
            images = batch['image'].to(device)
            captions = batch['caption']
            batch_size = images.size(0)

            t = torch.randint(0, config['timesteps'], (batch_size,), device=device).long()
            
            
            with torch.no_grad():
                
                z_0 = VAE_model.encode(images) 
                
                text_embeddings = text_encoder.encode_pooled(captions)
            

            
            noise = torch.randn_like(z_0)
            
            z_t = Diffusion_model.q_xt_given_x0(z_0, t, noise=noise)

            optimizer.zero_grad()
            
            with autocast(enabled= config.get('mixed_precision', True)):
                
                predicted_noise = Denoising_model(z_t, t, context=text_embeddings)

                
                mse_loss = nn.functional.mse_loss(predicted_noise, noise)
                
                
                
        
            scaler.scale(mse_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(Denoising_model.parameters(), max_grad_norm = config.get('max_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(Denoising_model)
            iteration += 1

            if iteration % config['print_frequency'] == 0:
                print(f"Epoch [{epoch}], Iteration [{iteration}/{total_iterations}] , Loss: {mse_loss.item():.4f}")
                    
            if iteration % config['save_frequency'] == 0:
                ema.apply_shadow(Denoising_model)
                torch.save(Denoising_model.state_dict(), 
                        os.path.join(config['checkpoint_dir'], f"model_iteration_{iteration} .pth"))
                ema.restore(Denoising_model)
                
            if iteration % config['sample_frequency'] == 0:
                ema.apply_shadow(Denoising_model)

                
            if iteration >= total_iterations:
                break
            
            
        epoch += 1
            
            
        
        
        
    print("Training Done !!!!!")


def sample_images(model, config , device):
    num_samples = config['num_samples']
    model.eval()
    with torch.no_grad():
        sampled_images = model.sample((num_samples, 3, 256, 256), device=device) 
    return sampled_images


    
    
