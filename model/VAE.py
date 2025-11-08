# vae.py
import torch
import torch.nn as nn
import torch.nn.functional as F



class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)

class Downsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or in_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.nin_shortcut = None
        if in_channels != out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        if self.nin_shortcut is not None:
            x = self.nin_shortcut(x)
        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        b, c, h_, w_ = q.shape
        q = q.reshape(b, c, h_ * w_)
        k = k.reshape(b, c, h_ * w_)
        v = v.reshape(b, c, h_ * w_)

        attn = torch.bmm(q.transpose(1, 2), k) * (c ** -0.5)
        attn = F.softmax(attn, dim=2)
        out = torch.bmm(v, attn.transpose(1, 2))
        out = out.reshape(b, c, h_, w_)
        out = self.proj_out(out)
        return x + out



class Encoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, attn_resolutions=(16,)):
        super().__init__()
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.conv_in = nn.Conv2d(3, ch, kernel_size=3, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if (32 // (2**i_level)) in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            else:
                down.downsample = None
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, 8, kernel_size=3, padding=1)  

    def forward(self, x):
        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
            if self.down[i_level].downsample is not None:
                h = self.down[i_level].downsample(h)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h

class Decoder(nn.Module):
    def __init__(self, ch=128, out_ch=3, ch_mult=(1, 2, 4, 4), num_res_blocks=2, attn_resolutions=(16,)):
        super().__init__()
        block_in = ch * ch_mult[-1]
        self.conv_in = nn.Conv2d(4, block_in, kernel_size=3, padding=1) 

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(len(ch_mult))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(num_res_blocks + 1):
                block.append(ResnetBlock(block_in, block_out))
                block_in = block_out
                if (32 // (2**i_level)) in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            else:
                up.upsample = None
            self.up.insert(0, up)  

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, padding=1)

    def forward(self, z):
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        for i_level in reversed(range(len(self.up))):
            for i_block in range(len(self.up[i_level].block)):
                h = self.up[i_level].block[i_block](h)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if self.up[i_level].upsample is not None:
                h = self.up[i_level].upsample(h)
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)
        return h



class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim=4):
        super().__init__()
        self.encoder = Encoder(ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2)
        self.decoder = Decoder(ch=128, ch_mult=[1, 2, 4, 4], num_res_blocks=2)
        self.quant_conv = nn.Conv2d(8, 2 * embed_dim, kernel_size=1)  
        self.post_quant_conv = nn.Conv2d(embed_dim, 4, kernel_size=1)
        self.embed_dim = embed_dim

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h) 
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z):
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, sample_posterior=True):
        mean, logvar = self.encode(x)
        if sample_posterior:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        recon = self.decode(z)
        return recon, mean, logvar


