import torch
import torch.nn as nn
import torch.nn.functional as F



def get_timestep_embedding(timesteps, embedding_dim):
    
    half_dim = embedding_dim // 2
    emb = torch.exp(
        -torch.log(torch.tensor(10000.0)) * 
        torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1)
    ).to(timesteps.device)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb



class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = F.silu(self.linear2(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # time conditioning
        time_mod = self.time_proj(F.silu(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_mod
        
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, context_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        
        # Self attention
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
        # Cross attention -  text conditioning
        self.text_norm = nn.LayerNorm(channels)
        self.text_proj = nn.Linear(context_dim, channels)
        self.text_q = nn.Linear(channels, channels)
        self.text_k = nn.Linear(context_dim, channels)
        self.text_v = nn.Linear(context_dim, channels)
        self.text_out = nn.Linear(channels, channels)

    def forward(self, x, context):
        B, C, H, W = x.shape
        h = self.norm(x)
        
        # Self attention
        qkv = self.qkv(h).reshape(B, C * 3, H * W).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        
        attn = torch.softmax(torch.bmm(q, k.transpose(-1, -2)) / (C ** 0.5), dim=-1)
        self_out = torch.bmm(attn, v).transpose(1, 2).reshape(B, C, H, W)
        h = x + self.proj(self_out)
        
        # Cross-attention ( for text context )
        if context is not None:
            text_features = self.text_norm(h.reshape(B, C, H * W).transpose(1, 2))  
            text_k = self.text_k(context)  
            text_v = self.text_v(context) 
            
            text_q_out = self.text_q(text_features)  
            text_attn = torch.softmax(
                torch.bmm(text_q_out, text_k.transpose(-1, -2)) / (C ** 0.5), 
                dim=-1
            )
            text_out = torch.bmm(text_attn, text_v).transpose(1, 2).reshape(B, C, H, W)
            h = h + self.text_out(text_out.transpose(1, 2).reshape(B, C, H, W))
        
        return h

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)

class DownsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    def __init__(self, in_channels=4, context_dim=768):
        super().__init__()
        
        # Time embedding
        self.time_mlp = TimestepEmbedding(320, 1280)
        
        
        self.input_proj = nn.Conv2d(in_channels, 320, 3, padding=1)
        
        
        self.encoder_layers = nn.ModuleList([
            #  320 channels
            nn.Sequential(
                ResidualBlock(320, 320, 1280),
                AttentionBlock(320, context_dim),
                ResidualBlock(320, 320, 1280),
                AttentionBlock(320, context_dim)
            ),
            
            #  320 -> 640
            nn.Sequential(
                DownsampleBlock(320),
                ResidualBlock(320, 640, 1280),
                AttentionBlock(640, context_dim),
                ResidualBlock(640, 640, 1280),
                AttentionBlock(640, context_dim)
            ),
            
            #  640 -> 1280
            nn.Sequential(
                DownsampleBlock(640),
                ResidualBlock(640, 1280, 1280),
                AttentionBlock(1280, context_dim),
                ResidualBlock(1280, 1280, 1280),
                AttentionBlock(1280, context_dim)
            ),
            
            #  1280 -> 1280
            nn.Sequential(
                DownsampleBlock(1280),
                ResidualBlock(1280, 1280, 1280),
                ResidualBlock(1280, 1280, 1280)
            )
        ])
        
        
        self.bottleneck = nn.Sequential(
            ResidualBlock(1280, 1280, 1280),
            AttentionBlock(1280, context_dim),
            ResidualBlock(1280, 1280, 1280)
        )
        
        # Decoder 
        self.decoder_layers = nn.ModuleList([
            # 1280 -> 1280
            nn.Sequential(
                ResidualBlock(2560, 1280, 1280),  # 1280 + 1280 from skip
                ResidualBlock(2560, 1280, 1280),
                UpsampleBlock(1280)
            ),
            
            # 2560 -> 1280 -> 640
            nn.Sequential(
                ResidualBlock(2560, 1280, 1280),
                AttentionBlock(1280, context_dim),
                ResidualBlock(2560, 1280, 1280),
                AttentionBlock(1280, context_dim),
                UpsampleBlock(1280)
            ),
            
            #  1920 -> 640 -> 320
            nn.Sequential(
                ResidualBlock(1920, 640, 1280),
                AttentionBlock(640, context_dim),
                ResidualBlock(1280, 640, 1280),
                AttentionBlock(640, context_dim),
                UpsampleBlock(640)
            ),
            
            #  960 -> 320 -> 320
            nn.Sequential(
                ResidualBlock(960, 320, 1280),
                AttentionBlock(320, context_dim),
                ResidualBlock(640, 320, 1280),
                AttentionBlock(320, context_dim),
                ResidualBlock(640, 320, 1280)
            )
        ])
        
        
        self.output_norm = nn.GroupNorm(32, 320)
        self.output_conv = nn.Conv2d(320, in_channels, 3, padding=1)

    def forward(self, x, timesteps, context):
        
        t_emb = get_timestep_embedding(timesteps, 320)
        t_emb = self.time_mlp(t_emb)
        
        
        x = self.input_proj(x)
        
        
        skips = []
        

        for level in self.encoder_layers:
            x = level(x) if isinstance(level, nn.Sequential) else level(x)
            skips.append(x)
        
        
        x = self.bottleneck(x)
        
        
        for i, level in enumerate(self.decoder_layers):
            skip = skips.pop()  
            x = torch.cat([x, skip], dim=1)  
            x = level(x)
        
        x = F.silu(self.output_norm(x))
        x = self.output_conv(x)
        
        return x


class DenoisingModel(nn.Module):
    def __init__(self, in_channels=4, context_dim=768):
        super().__init__()
        self.unet = UNet(in_channels, context_dim)

    def forward(self, x, timesteps, context):
        
        # x - [B, 4, H, W] 
        # timesteps - [B] 
        # context - [B, 77, 768] 
        
        return self.unet(x, timesteps, context)

