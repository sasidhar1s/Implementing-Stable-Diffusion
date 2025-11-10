import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
import json
from data.dataset import DiffusionDataset , ValidationDataset
import os
from DiffusionModule import Diffusion
from model.clip_import import  CLIPTextEncoder
import math
from torch.cuda.amp import autocast, GradScaler
from model.VAE_import import StableDiffusionVAE as VAE
from model.Unet import DenoisingModel 
from PIL import Image


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

    sampling_dataset = ValidationDataset(prompts_file=config['validation_prompts_file'])
    sampling_dataloader = DataLoader(sampling_dataset, batch_size= config['sampling_batch_size'], shuffle=False)
    
    Diffusion_model = Diffusion(
        timesteps=config['timesteps'],device=device,
        beta_start=config['beta_start'],
        beta_end=config['beta_end'] )
    
    
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
        
        Denoising_model.train()
        

        for batch in (dataloader):
            
            images = batch['image'].to(device)
            captions = batch['caption']
            batch_size = images.size(0)

            t = torch.randint(0, config['timesteps'], (batch_size,), device=device).long()
            
            
            with torch.no_grad():
                
                z_0 = VAE_model.encode(images) 
                
                text_embeddings = text_encoder.encode_last_hidden_state(captions)
            

            
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




def generate_validation_samples(diffusion_model, denoising_model, text_encoder, vae_model, 
                            validation_prompts, config, device, iteration):
    
    denoising_model.eval()
    
    os.makedirs(os.path.join(config['checkpoint_dir'], 'samples'), exist_ok=True)  # Fixed: was Validation_dir
    
    with torch.no_grad():
        
        cfg_scale = config.get('cfg_scale', 1.0)
        

        cond_embeddings = text_encoder.encode_last_hidden_state(validation_prompts)  # [N, 77, 768]
        uncond_embeddings = text_encoder.get_unconditional_embeddings(len(validation_prompts))  # [N, 77, 768]
        

        sampled_latents = sample_with_cfg(
            diffusion_model, denoising_model, 
            cond_embeddings, uncond_embeddings, 
            cfg_scale, config, device
        )
        
        
        sampled_latents = sampled_latents / 0.18215  # normalizing VAE latents by dividing with varaince
        sampled_images = vae_model.decode(sampled_latents)  # [N, 3, H, W]
        
        
        for prompt_idx, (img, prompt) in enumerate(zip(sampled_images, validation_prompts)):
            
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            
            pil_img = Image.fromarray((img.cpu().numpy() * 255).astype('uint8'))
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            pil_img.save(os.path.join(config['checkpoint_dir'], 'samples', 
                                    f"sample_iter_{iteration}_cfg_{cfg_scale}_prompt_{prompt_idx}_{safe_prompt[:50]}.png"))
    
    print(f"Validation samples saved for iteration {iteration}")

def sample_with_cfg(diffusion_model, denoising_model, cond_context, uncond_context, 
                cfg_scale, config, device):

    num_samples = cond_context.shape[0]
    latent_shape = (num_samples, config['latent_channels'], config['latent_size'], config['latent_size'])
    
    latents = torch.randn(latent_shape, device=device)
    
    for i in reversed(range(diffusion_model.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        latents_batch = latents.repeat(2, 1, 1, 1)
        t_batch = t.repeat(2)
        context_batch = torch.cat([uncond_context, cond_context], dim=0)
        
        noise_pred = denoising_model(latents_batch, t_batch, context_batch)
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        if cfg_scale == 1.0:
            noise_pred_cfg = noise_pred_cond  # No CFG effect
        else:
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        
        latents = diffusion_model.p_xtminus1_given_xt(latents, t, denoising_model, context=None)
    
    return latents
# update the sampling function to include cfg