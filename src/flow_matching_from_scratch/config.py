from dataclasses import dataclass


@dataclass
class UNetConfig:
    in_channels: int = 3
    out_channels: int = 3
    image_size: int = 32
    block_out_channels: tuple = (128, 256, 256)
    time_embd_dim: int = 512
    num_groups: int = 32
    kernel_size: int = 3
    padding: int = 1
    dropout: float = 0.0
    layers_per_block: int = 2


@dataclass
class TrainConfig:
    # data
    dataset: str = "CIFAR10"
    num_workers_train: int = 4
    num_workers_val: int = 2
    use_pin_memory: bool = True

    # batch
    total_batch_size: int = 256
    batch_size: int = 128

    # flow matching
    vf_model: str = "OT"
    sigma_min: float = 0.01

    # optimization
    max_lr: float = 2e-4
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.0
    max_steps: int = 300000
    grad_clip: float = 1.0
    warmup_steps: int = 3000
    use_polynomial: bool = True
    lr_power: float = 2.0

    # eval
    eval_every: int = 3000
    eval_steps: int = 100

    # generate
    generate_every: int = 3000
    num_generated: int = 16
    euler_steps: int = 100

    # logging
    project: str = "flow-matching-from-scratch"
    name: str = "cifar10-ot-baseline"
    group: str = "cifar10"
    use_wandb: str = False
    seed: int = 42

    @property
    def min_lr(self) -> float:
        return self.max_lr * self.min_lr_ratio

    def grad_accum_steps(self, ddp_world_size: int) -> int:
        assert self.total_batch_size % (self.batch_size * ddp_world_size) == 0
        return self.total_batch_size // (self.batch_size * ddp_world_size)

    def get_lr(self, it: int) -> float:
        if self.use_polynomial == False:
            return self.max_lr
        if it < self.warmup_steps:
            return self.max_lr * (it + 1) / self.warmup_steps
        if it >= self.max_steps:
            return self.min_lr
        base = (it - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.min_lr + (self.max_lr - self.min_lr) * (1 - base) ** self.lr_power
