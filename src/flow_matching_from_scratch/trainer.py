import time

from itertools import islice

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch import distributed as dist

from flow_matching_from_scratch.data import build_loader, infinite_loader
from flow_matching_from_scratch.model import UNet2DModel
from flow_matching_from_scratch.generate import save_images
import wandb, weave
from dataclasses import asdict


class Trainer:
    def __init__(self, unet_config, config, info):
        self.unet_config = unet_config
        self.config = config
        self.info = info
        self.grad_accum_steps = config.grad_accum_steps(self.info.ddp_world_size)
        self.train_loader = infinite_loader(
            build_loader(
                config,
                "train",
                info.ddp_world_size,
                info.ddp_rank,
                num_workers=config.num_workers_train,
                shuffle=True,
            )
        )
        self.val_loader = build_loader(
            config,
            "split",
            info.ddp_world_size,
            info.ddp_rank,
            num_workers=config.num_workers_val,
            shuffle=False,
        )
        self.model = UNet2DModel(unet_config)
        self.model.to(info.device)
        self.base_model = self.model
        self.model = torch.compile(self.model)
        if info.ddp:
            self.model = DDP(self.model, device_ids=[info.ddp_local_rank])
        self.optimizer = self.base_model.configure_optimizer(
            weight_decay=config.weight_decay,
            lr=config.max_lr,
            device=info.device,
            master_process=info.master_process,
        )

        if info.master_process and config.use_wandb:
            wandb.init(
                project=config.project,
                name=config.name,
                group=config.group,
                config={
                    **asdict(unet_config),
                    **asdict(config),
                    "min_lr": config.min_lr,
                    "grad_accum_steps": self.grad_accum_steps,
                    "world_size": info.ddp_world_size,
                },
            )
            weave.init(config.project)

    def build_cvf(self, x, x1, t):
        if self.config.vf_model == "OT":
            u = (x1 - (1 - self.config.sigma_min) * x) / (
                1 - (1 - self.config.sigma_min) * t
            )
            return u

    @torch.no_grad()
    def evaluate(self, step):
        self.model.eval()
        val_loss_acum = torch.zeros((), device=self.info.device)
        val_loss_steps = min(self.config.eval_steps, len(self.val_loader))
        g = torch.Generator(device=self.info.device)
        g.manual_seed(self.config.seed + self.info.ddp_rank)
        for x1, y in islice(self.val_loader, val_loss_steps):
            x1, y = x1.to(self.info.device), y.to(self.info.device)
            B, C, H, W = x1.size()
            t = torch.rand(B, device=self.info.device, generator=g)
            t4 = t[:, None, None, None]
            mu = t4 * x1
            sigma = 1 - (1 - self.config.sigma_min) * t4
            x0 = torch.randn((B, C, H, W), device=self.info.device, generator=g)
            x = mu + x0 * sigma
            u = self.build_cvf(x, x1, t4)
            with torch.autocast(
                device_type=self.info.device_type, dtype=torch.bfloat16
            ):
                pred, loss = self.model(x, t * 1000, target=u)
            loss = loss / val_loss_steps
            val_loss_acum += loss.detach()
        if self.info.ddp:
            dist.all_reduce(val_loss_acum, op=dist.ReduceOp.AVG)
        if self.info.master_process:
            print(f"validation loss: {val_loss_acum.item():.4f}")
            if self.config.use_wandb:
                wandb.log({"val/loss": val_loss_acum.item()}, step=step)

    def train_step(self, step):
        t0 = time.time()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_accum = 0
        for micro_step in range(self.grad_accum_steps):
            x1, y = next(self.train_loader)
            x1, y = x1.to(self.info.device), y.to(self.info.device)
            B, C, H, W = x1.size()
            assert (C, H, W) == (
                self.unet_config.in_channels,
                self.unet_config.image_size,
                self.unet_config.image_size,
            )

            t = torch.rand(B, device=self.info.device)
            t4 = t[:, None, None, None]
            mu = t4 * x1
            sigma = 1 - (1 - self.config.sigma_min) * t4
            x0 = torch.randn((B, C, H, W), device=self.info.device)
            x = mu + x0 * sigma
            u = self.build_cvf(x, x1, t4)
            with torch.autocast(
                device_type=self.info.device_type, dtype=torch.bfloat16
            ):
                pred, loss = self.model(x, t * 1000, target=u)
            loss = loss / self.grad_accum_steps
            loss_accum += loss.detach()
            if self.info.ddp:
                self.model.require_backward_grad_sync = (
                    micro_step == self.grad_accum_steps - 1
                )
            loss.backward()
        if self.info.ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )
        lr = self.config.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.optimizer.step()
        if self.info.device_type == "cuda":
            torch.cuda.synchronize()
        t1 = time.time()
        dt = (t1 - t0) * 1000
        images_processed = B * self.grad_accum_steps * self.info.ddp_world_size
        images_per_sec = images_processed / (t1 - t0)
        if self.info.master_process:
            print(
                f"step {step:4d} | loss {loss_accum.item():.6f} | lr {lr:.4e} | norm {norm:.4f}  | dt {dt:.2f} ms, images/s {images_per_sec:.2f}"
            )
            if self.config.use_wandb:
                wandb.log(
                    {
                        "train/loss": loss_accum.item(),
                        "train/lr": lr,
                        "train/grad_norm": norm.item(),
                        "train/images_per_sec": images_per_sec,
                        "train/dt_ms": dt,
                    },
                    step=step,
                )

    def fit(self):
        for step in range(self.config.max_steps):
            last_step = step == self.config.max_steps - 1

            if step % self.config.eval_every == 0 or last_step:
                self.evaluate(step)

            if (
                step % self.config.generate_every == 0 or last_step
            ) and self.info.master_process:
                save_images(
                    self.base_model,
                    self.config,
                    self.unet_config,
                    self.info.device,
                    step,
                    log_wandb=self.config.use_wandb,
                )

            self.train_step(step)
        if self.info.master_process:
            torch.save(
                self.base_model.state_dict(),
                f"cfm_ot_model_weights_{self.config.name}.pt",
            )
            print("weights saved")
            if self.config.use_wandb:
                wandb.finish()
