from torch import nn
import torch
from dataclasses import dataclass

from torch.nn.qat import Linear


class ResnetBlock2D(nn.Module):
    def __init__(self, config):
        super().__init__()


class AttentionBlock(nn.Module):
    def __init__(self, config):
        super().__init__()


class AttnDownBlock2D(nn.Module):
    def __init__(self, config):
        super().__init__()


class AttnUpBlock2D(nn.Module):
    def __init__(self, config):
        super().__init__()


class DownBlock2d(nn.Module):
    def __init__(self, config):
        super().__init__()


class UpBlock2d(nn.Module):
    def __init__(self, config):
        super().__init__()


class UNetMidBlock2D(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attentions = nn.ModuleList()


class UNet2DModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.conv_in = nn.Conv2d(
            in_channels=config.in_channels,
            out_channels=config.n_channel_1,
            kernel_size=config.kernel_size,
            padding=config.padding,
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(config.n_channels, config.time_embd_dim),
            nn.SiLU(),
            nn.Linear(config.time_embd_dim, config.time_embd_dim),
        )
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2d(config),
                AttnDownBlock2D(config),
                AttnDownBlock2D(config),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                AttnUpBlock2D(config),
                AttnUpBlock2D(config),
                UpBlock2d(config),
            ]
        )
        self.mid_block = UNetMidBlock2D(config)
        self.conv_norm_out = nn.GroupNorm(
            config.n_block_channels[0], eps=1e-05, affine=True
        )
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2D(
            in_channels=config.n_block_channels[0],
            out_channels=config.out_channels,
            kernel_size=config.kernel_size,
            padding=config.padding,
        )
