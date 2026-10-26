from torch import nn
from torch.nn import functional as F
import torch
import inspect


class Timesteps(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.num_channels = num_channels
        assert self.num_channels % 2 == 0
        freqs = torch.arange(0, self.num_channels // 2)
        freqs = 10000 ** (-1.0 * freqs / (self.num_channels // 2))  # (n_ch// 2,)
        self.register_buffer("freqs", freqs, persistent=False)


    def forward(self, t):
        # t (B,)
        freqs = self.freqs
        temp_embd = t[:, None] * freqs[None, :]  # (B, n_ch // 2)
        out = torch.cat([temp_embd.cos(), temp_embd.sin()], dim=1)  # (B, n_ch)
        return out


class TimestepEmbedding(nn.Module):
    def __init__(self, in_dim, time_embd_dim):
        super().__init__()
        self.linear_1 = nn.Linear(in_dim, time_embd_dim)
        self.act = nn.SiLU()
        self.linear_2 = nn.Linear(time_embd_dim, time_embd_dim)

    def forward(self, t):
        t = self.linear_1(t)
        t = self.act(t)
        out = self.linear_2(t)
        return out


class Downsample2D(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=config.kernel_size,
            stride=2,
            padding=config.padding,
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class Upsample2D(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=config.kernel_size,
            padding=config.padding,
        )

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2)
        out = self.conv(x)
        return out


class ResnetBlock2D(nn.Module):
    def __init__(self, config, in_channels, out_channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(config.num_groups, in_channels, eps=1e-5, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, config.kernel_size, padding=config.padding
        )
        self.time_embd_proj = nn.Linear(config.time_embd_dim, out_channels)
        self.norm2 = nn.GroupNorm(
            config.num_groups, out_channels, eps=1e-5, affine=True
        )
        self.dropout = nn.Dropout(config.dropout)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, config.kernel_size, padding=config.padding
        )
        self.nonlinearity = nn.SiLU()
        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv_shortcut = None

    def forward(self, x, t):
        h = self.nonlinearity(self.norm1(x))
        h = self.conv1(h)  # (B, out_ch, H, W)
        t = self.time_embd_proj(self.nonlinearity(t))  # (B, out_ch)
        h = h + t[:, :, None, None]
        h = self.nonlinearity(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        if self.conv_shortcut is not None:
            x = self.conv_shortcut(x)
        return h + x


class AttentionBlock(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(
            config.num_groups, channels, eps=1e-5, affine=True
        )
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.proj_attn = nn.Linear(channels, channels)

    def forward(self, x):
        B, C, H, W = x.size()
        h = self.group_norm(x).view(B, C, H * W).transpose(-1, -2)  # (B, HW, C)
        q = self.query(h)
        k = self.key(h)
        v = self.value(h)
        y = F.scaled_dot_product_attention(q, k, v)  # (B, HW, C)
        y = self.proj_attn(y)
        out = y.transpose(-1, -2).view(B, C, H, W)
        return x + out


class DownBlock2D(nn.Module):
    def __init__(self, config, in_channels, out_channels, add_downsample=True):
        super().__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    config, in_channels if i == 0 else out_channels, out_channels
                )
                for i in range(config.layers_per_block)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(config, out_channels)])
            if add_downsample
            else None
        )

    def forward(self, x, t):
        skips = []
        for resnet in self.resnets:
            x = resnet(x, t)
            skips.append(x)
        if self.downsamplers is not None:
            for dowsampler in self.downsamplers:
                x = dowsampler(x)
                skips.append(x)
        return x, skips


class AttnDownBlock2D(nn.Module):
    def __init__(self, config, in_channels, out_channels, add_downsample=True):
        super().__init__()
        self.attentions = nn.ModuleList(
            [
                AttentionBlock(config, out_channels)
                for _ in range(config.layers_per_block)
            ]
        )
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(
                    config, in_channels if i == 0 else out_channels, out_channels
                )
                for i in range(config.layers_per_block)
            ]
        )
        self.downsamplers = (
            nn.ModuleList([Downsample2D(config, out_channels)])
            if add_downsample
            else None
        )

    def forward(self, x, t):
        skips = []
        for resnet, attention in zip(self.resnets, self.attentions):
            x = resnet(x, t)
            x = attention(x)
            skips.append(x)
        if self.downsamplers is not None:
            for downsampler in self.downsamplers:
                x = downsampler(x)
                skips.append(x)
        return x, skips


class UpBlock2D(nn.Module):
    def __init__(self, config, prev_out, out_channels, skip_in, add_upsample=True):
        super().__init__()
        n = config.layers_per_block + 1
        self.resnets = nn.ModuleList()
        for i in range(n):
            skip = skip_in if i == n - 1 else out_channels
            res_in = prev_out if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(config, res_in + skip, out_channels))
        self.upsamplers = (
            nn.ModuleList([Upsample2D(config, out_channels)]) if add_upsample else None
        )

    def forward(self, x, t, skips):
        for resnet in self.resnets:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, t)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class AttnUpBlock2D(nn.Module):
    def __init__(self, config, prev_out, out_channels, skip_in, add_upsample=True):
        super().__init__()
        n = config.layers_per_block + 1
        self.attentions = nn.ModuleList(
            [AttentionBlock(config, out_channels) for _ in range(n)]
        )
        self.resnets = nn.ModuleList()
        for i in range(n):
            skip = skip_in if i == n - 1 else out_channels
            res_in = prev_out if i == 0 else out_channels
            self.resnets.append(ResnetBlock2D(config, res_in + skip, out_channels))
        self.upsamplers = (
            nn.ModuleList([Upsample2D(config, out_channels)]) if add_upsample else None
        )

    def forward(self, x, t, skips):
        for resnet, attention in zip(self.resnets, self.attentions):
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = resnet(x, t)
            x = attention(x)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                x = upsampler(x)
        return x


class UNetMidBlock2D(nn.Module):
    def __init__(self, config, channels):
        super().__init__()
        self.attentions = nn.ModuleList([AttentionBlock(config, channels)])
        self.resnets = nn.ModuleList(
            [ResnetBlock2D(config, channels, channels) for _ in range(2)]
        )

    def forward(self, x, t):
        x = self.resnets[0](x, t)
        x = self.attentions[0](x)
        x = self.resnets[1](x, t)
        return x


class UNet2DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        ch = list(config.block_out_channels)  # [32, 64, 64]
        n = len(ch)

        self.conv_in = nn.Conv2d(
            config.in_channels, ch[0], config.kernel_size, padding=config.padding
        )
        self.time_proj = Timesteps(ch[0])
        self.time_embedding = TimestepEmbedding(ch[0], config.time_embd_dim)

        # down : DownBlock2D, AttnDownBlock2D, AttnDownBlock2D
        self.down_blocks = nn.ModuleList()
        out_c = ch[0]
        for i in range(n):
            in_c = out_c
            out_c = ch[i]
            is_last = i == n - 1
            block = DownBlock2D if i == 0 else AttnDownBlock2D
            self.down_blocks.append(
                block(config, in_c, out_c, add_downsample=not is_last)
            )

        self.mid_block = UNetMidBlock2D(config, ch[-1])

        # up : AttnUpBlock2D, AttnUpBlock2D, UpBlock2D
        rev = list(reversed(ch))  # [64, 64, 32]
        self.up_blocks = nn.ModuleList()
        out_c = rev[0]
        for i in range(n):
            prev_out = out_c
            out_c = rev[i]
            skip_in = rev[min(i + 1, n - 1)]
            is_last = i == n - 1
            block = UpBlock2D if is_last else AttnUpBlock2D
            self.up_blocks.append(
                block(config, prev_out, out_c, skip_in, add_upsample=not is_last)
            )

        self.conv_norm_out = nn.GroupNorm(
            config.num_groups, ch[0], eps=1e-5, affine=True
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(
            ch[0], config.out_channels, config.kernel_size, padding=config.padding
        )

    def forward(self, x, t, target=None):
        t = self.time_embedding(self.time_proj(t))
        x = self.conv_in(x)
        skips = [x]
        for down_block in self.down_blocks:
            x, skips_block = down_block(x, t)
            skips.extend(skips_block)
        x = self.mid_block(x, t)
        for up_block in self.up_blocks:
            x = up_block(x, t, skips)
        assert len(skips) == 0
        x = self.conv_act(self.conv_norm_out(x))
        out = self.conv_out(x)
        loss = None
        if target is not None:
            loss = F.mse_loss(
                out, target
            )
        return out, loss

    def configure_optimizer(self, weight_decay, lr, device, master_process):
        param_dict = {n: p for n, p in self.named_parameters()}
        param_dict = {n: p for n, p in param_dict.items() if p.requires_grad}
        decay_param = [p for _, p in param_dict.items() if p.dim() >= 2]
        nodecay_param = [p for _, p in param_dict.items() if p.dim() < 2]
        optim_param = [
            {"params": decay_param, "weight_decay": weight_decay},
            {"params": nodecay_param, "weight_decay": 0.0},
        ]
        num_decay_param = sum(p.numel() for p in decay_param)
        num_nodecay_param = sum(p.numel() for p in nodecay_param)
        if master_process:
            print(f"num decay param: {len(decay_param)} with {num_decay_param}")
            print(f"num nodecay para: {len(nodecay_param)} with {num_nodecay_param}")
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and "cuda" in device
        if master_process:
            print("using fused in AdamW", use_fused)
        optimizer_adamw = torch.optim.AdamW(
            optim_param, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=use_fused
        )
        return optimizer_adamw
