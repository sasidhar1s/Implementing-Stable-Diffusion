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
from PIL import Image
import logging
from datetime import datetime
import json
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from utils import load_checkpoint
from utils import setup_distributed, cleanup_distributed, get_cosine_schedule_with_warmup, ExponentialMovingAverage_EMA




                
                
def train_model(rank, world_size, device_id, config):
    is_main_process = rank == 0
    exp_dir = None 

    if is_main_process:
        exp_dir = f"experiments/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
        
        os.makedirs(f"{exp_dir}/samples", exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{exp_dir}/training.log"),
                logging.StreamHandler()] )

        with open(f"{exp_dir}/config.json", 'w') as f:
            json.dump(config, f, indent=4)
    
    # Broadcasting the experiment file path directory to all processes
    if world_size > 1:
        
        exp_dir_list = [exp_dir] if is_main_process else [None]
        dist.broadcast_object_list(exp_dir_list, src=0)
        exp_dir = exp_dir_list[0]
        config['exp_dir'] = exp_dir 

    device = torch.device(f"cuda:{device_id}")

    dataset = DiffusionDataset(image_dir=config['image_dir'], captions_file=config['captions_file'], 
                            image_size=config['image_size'],
                            caption_dropout_prob=config['caption_dropout_prob'])
    
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    
    dataloader = DataLoader(dataset, batch_size=config['batch_size_per_gpu'], 
                            num_workers=8, pin_memory=True, sampler=sampler, shuffle=False)
    
    Diffusion_model = Diffusion(
        timesteps=config['timesteps'], device=device,
        beta_start=config['beta_start'],
        beta_end=config['beta_end'])
    
    Denoising_model = DenoisingModel(in_channels=config['latent_channels'],
                                    context_dim=config['context_dim']).to(device)
    
    text_encoder = CLIPTextEncoder(model_name=config['image&text_encoding_model'], device=device)
    
    VAE_model = VAE(model_name=config['vae_model'], device=device)
    
    if is_main_process:
        logging.info(f"Model parameters: {sum(p.numel() for p in Denoising_model.parameters()):,}")
    
    optimizer = optim.AdamW(Denoising_model.parameters(), lr=config['learning_rate'],
                            weight_decay=config.get('weight_decay', 1e-2),
                            betas=(0.9, 0.999))

    total_iterations = config['total_iterations']
    
    warmup_iterations = config.get('warmup_iterations', 0.1 * total_iterations)
    
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_iterations,
                                                num_training_steps=total_iterations)
    
    scaler = GradScaler(enabled=config.get('mixed_precision', True))
    
    ema = ExponentialMovingAverage_EMA(Denoising_model, decay=config.get('ema_decay', 0.9999))

    iteration, epoch = load_checkpoint(config.get("resume_checkpoint"),
                    Denoising_model, optimizer, scheduler, ema, scaler, device, is_main_process)
        
    Denoising_model = DDP(Denoising_model, device_ids=[device_id])
    
    while iteration < total_iterations:
        sampler.set_epoch(epoch)
        Denoising_model.train()

        for batch in dataloader:
            images = batch['image'].to(device, non_blocking=True)
            captions = batch['caption']
            batch_size = images.size(0)

            t = torch.randint(0, config['timesteps'], (batch_size,), device=device).long()
            
            with torch.no_grad():
                
                z_0 = VAE_model.encode(images) 
                
                text_embeddings = text_encoder.encode_last_hidden_state(captions)
            
            noise = torch.randn_like(z_0)
            
            z_t, _ = Diffusion_model.q_xt_given_x0(z_0, t, noise=noise)

            optimizer.zero_grad(set_to_none=True)
            
            with autocast(enabled=config.get('mixed_precision', True)):
                
                predicted_noise = Denoising_model(z_t, t, context=text_embeddings)
                
                mse_loss = nn.functional.mse_loss(predicted_noise, noise)
        
        
            scaler.scale(mse_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(Denoising_model.parameters(), max_grad_norm=config.get('max_grad_norm', 1.0))
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(Denoising_model)
            iteration += 1

            
            if is_main_process:
                if iteration % config['print_frequency'] == 0:
                    logging.info(f"Epoch [{epoch}], Iteration [{iteration}/{total_iterations}], Loss: {mse_loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
                        
                
                if iteration > 0 and iteration % config['save_frequency'] == 0:
                    ema.apply_shadow(Denoising_model)
                    save_path = os.path.join(exp_dir, 'checkpoints', f"model_iteration_{iteration}.pth")
                    torch.save({
                        'iteration': iteration, 'epoch': epoch,
                        'model_state_dict': Denoising_model.module.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'ema_state_dict': ema.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, save_path)
                    logging.info(f"Saved checkpoint to {save_path}")
                    ema.restore(Denoising_model)
                    
                if iteration % config['sample_frequency'] == 0:
                    logging.info(f"Generating validation samples at iteration {iteration}...")
                    ema.apply_shadow(Denoising_model)
                    generate_validation_samples(Diffusion_model, Denoising_model.module, text_encoder, VAE_model,
                                                config, device, iteration, exp_dir)
                    ema.restore(Denoising_model)

            if iteration >= total_iterations:
                break
        
        epoch += 1
    
    if is_main_process:
        logging.info("Training Done !!!!!")
        
        
        
        


def generate_validation_samples(diffusion_model, denoising_model, text_encoder, vae_model, config, device, iteration, exp_dir):
    denoising_model.eval()
    
    with open(config['validation_prompts_file'], 'r') as f:
        validation_prompts = json.load(f)
    
    sample_dir = os.path.join(exp_dir, 'samples') 
    
    with torch.no_grad():
        cfg_scale = config.get('cfg_scale', 1.0) 
        cond_embeddings = text_encoder.encode_last_hidden_state(validation_prompts)
        uncond_embeddings = text_encoder.get_unconditional_embeddings(len(validation_prompts))
        
        sampled_latents = sample_with_cfg(diffusion_model, denoising_model, cond_embeddings, uncond_embeddings, 
                                        cfg_scale, config, device, config.get('clip_denoised', False), config.get('repeat_noise', False))
        
        sampled_latents = sampled_latents / 0.18215
        sampled_images = vae_model.decode(sampled_latents)
        
        for prompt_idx, (img, prompt) in enumerate(zip(sampled_images, validation_prompts)):
            img = (img.clamp(-1, 1) + 1) / 2 # Normalize to [0, 1]
            
            
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(img_np)
            
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            pil_img.save(os.path.join(sample_dir, f"sample_iter_{iteration}_cfg_{cfg_scale}_prompt_{prompt_idx}_{safe_prompt[:50]}.png"))
    
    logging.info(f"Validation samples saved for iteration {iteration}")

def sample_with_cfg(diffusion_model, denoising_model, cond_context, uncond_context, 
                    cfg_scale, config, device, clip_denoised, repeat_noise):
    num_samples = cond_context.shape[0]
    
    
    latent_dim = config['image_size'] // 8
    latent_shape = (num_samples, config['latent_channels'], latent_dim, latent_dim)
    
    latents = torch.randn(latent_shape, device=device)
    
    for i in reversed(range(diffusion_model.timesteps)):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        
        with torch.no_grad(), autocast(enabled=config.get('mixed_precision', True)):
            latents_batch = latents.repeat(2, 1, 1, 1)
            t_batch = t.repeat(2)
            context_batch = torch.cat([uncond_context, cond_context], dim=0)
            
            noise_pred = denoising_model(latents_batch, t_batch, context_batch)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        if cfg_scale == 1.0:
            noise_pred_cfg = noise_pred_cond
        else:
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        latents = diffusion_model.px_t_minus1_given_predicted_noise_and_x_t(noise_pred_cfg, latents, t, clip_denoised, repeat_noise)
    
    return latents


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--gpu_ids", type=str, required=True, help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3')")
    args = parser.parse_args()

    gpu_ids = [int(id_str) for id_str in args.gpu_ids.split(',')]
    
    with open(args.config, 'r') as f:
        config = json.load(f)

    
    rank, world_size, device_id = setup_distributed(gpu_ids)
    
    try:
        
        train_model(rank, world_size, device_id, config)
    finally:
        
        cleanup_distributed()