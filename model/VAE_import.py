import torch
from diffusers import AutoencoderKL
from PIL import Image
import torchvision.transforms as T

class StableDiffusionVAE:
    def __init__(self, model_name="stabilityai/stable-diffusion-2-1", device='cuda'):
       
        self.device = device
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder="vae")
        self.vae.to(device)
        self.vae.eval() 
        self.scaling_factor = self.vae.config.scaling_factor

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        
        
        images = images.to(self.device)
        
        with torch.no_grad():
            
            latent_dist = self.vae.encode(images).latent_dist
            
            
            latents = latent_dist.sample()
            
        
        latents = latents * self.scaling_factor
        
        return latents

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
      
      
        latents = latents.to(self.device)
        
        
        latents = latents / self.scaling_factor
        
        with torch.no_grad():
            
            images = self.vae.decode(latents).sample
            
        return images
