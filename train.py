import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
from data.dataset import DiffusionDataset 
import os
from transformers import CLIPModel, CLIPProcessor
from DiffusionModule import Diffusion
from model.clip_import import CLIPImageEncoder, CLIPTextEncoder, CLIPLoss 


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def train_model(config):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    dataset = DiffusionDataset(image_dir=config['image_dir'], text_file=config['text_file'])  
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4)

    
    
    model = Diffusion(
        timesteps=config['timesteps'],
        beta_start=config['beta_start'],
        beta_end=config['beta_end'] ).to(device)
    

    text_encoder = CLIPTextEncoder().to(device)
    image_encoder = CLIPImageEncoder().to(device)
    clip_loss_fn = CLIPLoss().to(device)
    


    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])


    for epoch in range(config['epochs']):
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


    
    
