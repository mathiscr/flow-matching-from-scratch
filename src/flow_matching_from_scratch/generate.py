import torch
from torchvision.utils import save_image
import os
import wandb
from PIL import Image

@torch.no_grad()
def generate_images(model, n, unet_config, steps=100, device="cuda"):
    model.eval()
    dt = 1.0 / steps
    c = unet_config
    assert c.in_channels == c.out_channels
    x = torch.randn((n, c.in_channels, c.image_size, c.image_size), device=device)
    x0 = x.clone()
    flow1 = flow2 = None
    for i in range(steps):
        t = torch.full((n,), fill_value=i * dt, device=device)
        v, _ = model(x, t * 1000)
        x = x + v * dt
        if steps // 3 == i + 1:
            flow1 = x.clone()
        if 2 * steps // 3 == i + 1:
            flow2 = x.clone()
    return x0, flow1, flow2, x


def make_grid(model, config, unet_config, device):
    n = config.num_generated
    x0, f1, f2, x1 = generate_images(
        model, n, unet_config, config.euler_steps, device=device
    )
    grid = torch.cat([x0, f1, f2, x1])
    return ((grid + 1) / 2).clamp(0, 1)


def save_images(model, config, unet_config, device, step=None, log_wandb=False):
    grid = make_grid(model, config, unet_config, device)
    n = config.num_generated
    os.makedirs("res", exist_ok=True)
    path = f"res/trajectory_{config.name}.png"
    save_image(grid, path, nrow=n)
    if log_wandb:
        wandb.log(
            {
                "samples/trajectory": wandb.Image(
                    Image.open(path), caption=f"step {step} | lines: noise, t=1/3, t=2/3, final"
                )
            },
            step=step,
        )
