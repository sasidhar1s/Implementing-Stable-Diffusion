import torch
import torch.nn as nn
import torch.nn.functional as F



def get_timestep_embedding(timesteps, embedding_dim):
    half_dim = embedding_dim // 2
    emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1) 
    emb = torch.exp(-emb * torch.arange(half_dim, device=timesteps.device))
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1: # zero pad
        emb = F.pad(emb, (0, 1))
    return emb

class TimestepEmbedding(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        x = F.silu(self.linear1(x))
        x = self.linear2(x) 
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, time_emb):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        time_mod = self.time_proj(F.silu(time_emb)).unsqueeze(-1).unsqueeze(-1)
        h = h + time_mod
        
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        
        return h + self.skip(x)

class AttentionBlock(nn.Module):
    def __init__(self, channels, context_dim=None, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.to_qkv = nn.Linear(channels, channels * 3)  
        self.to_out_self = nn.Linear(channels, channels)
        
        
        self.use_cross_attn = context_dim is not None
        if self.use_cross_attn:
            self.norm_cross = nn.GroupNorm(32, channels, eps=1e-6) 
            self.to_q_cross = nn.Linear(channels, channels)  
            self.to_kv_cross = nn.Linear(context_dim, channels * 2)  
            self.to_out_cross = nn.Linear(channels, channels)

    def forward(self, x, context=None):
        B, C, H, W = x.shape
        x_in = x
        
        
        x_norm = self.norm(x)
        x_norm = x_norm.view(B, C, H * W).transpose(1, 2)
        
        qkv = self.to_qkv(x_norm).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2), 
            qkv
        )
        
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).contiguous().view(B, H * W, C)
        attn = self.to_out_self(attn)
        attn = attn.transpose(1, 2).view(B, C, H, W)
        
        x = x_in + attn  
        
        
        if self.use_cross_attn and context is not None:
            x_norm = self.norm_cross(x)
            x_norm = x_norm.view(B, C, H * W).transpose(1, 2)
            
            q = self.to_q_cross(x_norm)
            kv = self.to_kv_cross(context)
            k, v = kv.chunk(2, dim=-1)
            
            q, k, v = map(
                lambda t: t.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2),
                (q, k, v)
            )
            
            attn = F.scaled_dot_product_attention(q, k, v)
            attn = attn.transpose(1, 2).contiguous().view(B, H * W, C)
            attn = self.to_out_cross(attn)
            attn = attn.transpose(1, 2).view(B, C, H, W)
            
            x = x + attn  
        
        return x

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




class UNetEncoderLevel(nn.Module):
    def __init__(self, modules):
        super().__init__()
        
        self.layers = nn.ModuleList(modules)

    def forward(self, x, t_emb, context):
        
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNetDecoderLevel(nn.Module):
    def __init__(self, modules):
        super().__init__()
        
        self.layers = nn.ModuleList(modules)

    def forward(self, x, t_emb, context):
        
        for layer in self.layers:
            if isinstance(layer, ResidualBlock):
                x = layer(x, t_emb)
            elif isinstance(layer, AttentionBlock):
                x = layer(x, context)
            else:
                x = layer(x)
        return x

class UNet(nn.Module):
    def __init__(self, in_channels=4, context_dim=768):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(320, 1280),
            nn.SiLU(),
            nn.Linear(1280, 1280)
        )
        
        self.input_proj = nn.Conv2d(in_channels, 320, 3, padding=1)
        
        self.encoder_layers = nn.ModuleList([
            UNetEncoderLevel([ResidualBlock(320, 320, 1280), 
                            AttentionBlock(320, context_dim), 
                            ResidualBlock(320, 320, 1280), 
                            AttentionBlock(320, context_dim)]),
            
            UNetEncoderLevel([DownsampleBlock(320),
                            ResidualBlock(320, 640, 1280), 
                            AttentionBlock(640, context_dim),
                            ResidualBlock(640, 640, 1280),
                            AttentionBlock(640, context_dim)]),
            
            UNetEncoderLevel([DownsampleBlock(640),
                            ResidualBlock(640, 1280, 1280),
                            AttentionBlock(1280, context_dim), 
                            ResidualBlock(1280, 1280, 1280), 
                            AttentionBlock(1280, context_dim)]),
            
            UNetEncoderLevel([DownsampleBlock(1280),
                            ResidualBlock(1280, 1280, 1280),
                            ResidualBlock(1280, 1280, 1280)])
        ])
        
        self.bottleneck = UNetEncoderLevel([
            ResidualBlock(1280, 1280, 1280),
            AttentionBlock(1280, context_dim),
            ResidualBlock(1280, 1280, 1280)
        ])
        
        self.decoder_layers = nn.ModuleList([
            UNetDecoderLevel([ResidualBlock(2560, 1280, 1280),
                            ResidualBlock(1280, 1280, 1280),
                            UpsampleBlock(1280)]),
            UNetDecoderLevel([ResidualBlock(2560, 1280, 1280),
                            AttentionBlock(1280, context_dim), 
                            ResidualBlock(1280, 1280, 1280), 
                            AttentionBlock(1280, context_dim), 
                            UpsampleBlock(1280)]),
            UNetDecoderLevel([ResidualBlock(1920, 640, 1280),
                            AttentionBlock(640, context_dim), 
                            ResidualBlock(640, 640, 1280), 
                            AttentionBlock(640, context_dim), 
                            UpsampleBlock(640)]),
            UNetDecoderLevel([ResidualBlock(960, 320, 1280), 
                            AttentionBlock(320, context_dim),
                            ResidualBlock(320, 320, 1280), 
                            AttentionBlock(320, context_dim),
                            ResidualBlock(320, 320, 1280)])
        ])
        
        self.output_norm = nn.GroupNorm(32, 320, eps=1e-6)
        self.output_conv = nn.Conv2d(320, in_channels, 3, padding=1)

    def forward(self, x, timesteps, context):
        

        
        t_emb = get_timestep_embedding(timesteps, 320)
        t_emb = self.time_mlp(t_emb)
        
        x = self.input_proj(x)
        
        skips = []
        for level in self.encoder_layers:
            x = level(x, t_emb, context)
            skips.append(x)
        
        x = self.bottleneck(x, t_emb, context)
        
        for level in self.decoder_layers:
            skip = skips.pop()  
            x = torch.cat([x, skip], dim=1)  
            x = level(x, t_emb, context)
        
        x = F.silu(self.output_norm(x))
        x = self.output_conv(x)
        
        return x
        
class DenoisingModel(nn.Module):
    def __init__(self, in_channels=4, context_dim=768):
        super().__init__()
        self.unet = UNet(in_channels, context_dim)

    def forward(self, x, timesteps, context):
        return self.unet(x, timesteps, context)