import torch
import json
import os
from model.Unet import DenoisingModel
from model.clip_import import CLIPTextEncoder
from model.VAE_import import StableDiffusionVAE as VAE
from DiffusionModule import Diffusion
from PIL import Image
import argparse
from tqdm import tqdm
from torch.amp import autocast



def sample_with_cfg(diffusion_model, denoising_model, cond_context, uncond_context, 
                    cfg_scale, config, device, clip_denoised, seed=None):
    num_samples = cond_context.shape[0]
    
    
    generator = torch.Generator(device=device)
    if seed:
        generator.manual_seed(seed)
    
    latent_dim = config['model_parameters']['image_size'] // 8
    latent_shape = (num_samples, config['model_parameters']['latent_channels'], latent_dim, latent_dim)
    
    latents = torch.randn(latent_shape, device=device, generator=generator)
    
    
    for i in tqdm(reversed(range(diffusion_model.timesteps)), desc="Sampling", total=diffusion_model.timesteps):
        t = torch.full((num_samples,), i, device=device, dtype=torch.long)
        
        with torch.no_grad(), autocast(device_type="cuda", enabled=True):
            latents_batch = latents.repeat(2, 1, 1, 1)
            t_batch = t.repeat(2)
            context_batch = torch.cat([uncond_context, cond_context], dim=0)
            
            noise_pred = denoising_model(latents_batch, t_batch, context_batch)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        
        if cfg_scale == 1.0:
            noise_pred_cfg = noise_pred_cond
        else:
            noise_pred_cfg = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
        
        
        latents = diffusion_model.px_t_minus1_given_predicted_noise_and_x_t(
            noise_pred_cfg, latents, t, clip_denoised
        )
    
    return latents


def run_inference(args):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    checkpoint_dir = os.path.dirname(args.checkpoint_path)
    config_path = os.path.join(checkpoint_dir, "..", "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Training config not found at {config_path}.")
    
    with open(config_path, 'r') as f:
        config = json.load(f)

    
    iteration_num = os.path.basename(args.checkpoint_path).split('_')[-1].split('.')[0]
    result_subfolder = f"iter_{iteration_num}"
    output_dir = os.path.join("results", result_subfolder)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    
    print("Loading models...")
    denoising_model = DenoisingModel(
        in_channels=config['model_parameters']['latent_channels'],
        context_dim=config['model_parameters']['context_dim']
    ).to(device)
    
    state_dict = torch.load(args.checkpoint_path, map_location=device)['model_state_dict']
    denoising_model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()})
    denoising_model.eval()

    text_encoder = CLIPTextEncoder(model_name=config['model_parameters']['image&text_encoding_model'], device=device)
    vae_model = VAE(model_name=config['model_parameters']['vae_model'], device=device)
    diffusion_model = Diffusion(
        timesteps=config['diffusion_schedule']['timesteps'], 
        device=device,
        beta_start=config['diffusion_schedule']['beta_start'],
        beta_end=config['diffusion_schedule']['beta_end'])

    
    with open(args.prompts_path, 'r') as f:
        prompts = json.load(f)
    print(f"Found {len(prompts)} prompts.")

   
    with torch.no_grad():
        uncond_embeddings = text_encoder.get_unconditional_embeddings(len(prompts))
        cond_embeddings = text_encoder.encode_last_hidden_state(prompts)
        
        print(f"Generating latents with full {diffusion_model.timesteps} steps and CFG scale {args.cfg_scale}...")
        sampled_latents = sample_with_cfg(
            diffusion_model, denoising_model, cond_embeddings, uncond_embeddings, 
            args.cfg_scale, config, device, 
            args.clip_denoised, args.seed
        )
        
        print("Decoding latents into images...")
        sampled_latents = sampled_latents 
        sampled_images = vae_model.decode(sampled_latents)
        
        print("Saving images...")
        for i, (img_tensor, prompt) in enumerate(zip(sampled_images, prompts)):
            img = (img_tensor.clamp(-1, 1) + 1) / 2
            img_np = (img.permute(1, 2, 0).cpu().numpy() * 255).astype('uint8')
            pil_img = Image.fromarray(img_np)
            
            safe_prompt = "".join(c for c in prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"iter_{iteration_num}_cfg_{args.cfg_scale}_seed_{args.seed}_{i:03d}_{safe_prompt[:100]}.png"
            pil_img.save(os.path.join(output_dir, filename))

    print("Inference complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate images from a trained diffusion model checkpoint.")
    
    
    parser.add_argument("--checkpoint_path", type=str, 
                        default='/Data2/sasi/StableDiffusion/experiments/twofiftysix_20251115_152451/checkpoints/model_iteration_240000.pth', help="Path to the model checkpoint (.pth file).")
    parser.add_argument("--prompts_path", type=str,
                        default='/Data2/sasi/combined_dataset/validation_prompts.json', help="Path to a JSON file containing a list of prompts.")
    
    
    parser.add_argument("--cfg_scale", type=float, default=3, help="Classifier-Free Guidance scale.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--clip_denoised", action='store_true', help="Enable clamping of the predicted x0.")

    args = parser.parse_args()
    run_inference(args)