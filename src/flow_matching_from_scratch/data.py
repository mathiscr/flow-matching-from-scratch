from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms


def build_loader(config, split, world_size, rank, num_workers=0, shuffle=True):
    is_train = split == "train"

    if config.dataset == "MNIST":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        dataset = datasets.MNIST(
            root="data", train=is_train, download=True, transform=transform
        )
    if config.dataset == "CIFAR10":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        dataset = datasets.CIFAR10(
            root="data", train=is_train, download=True, transform=transform
        )
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=is_train
    )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.use_pin_memory,
        drop_last=is_train,
        persistent_workers=num_workers > 0,
    )


def infinite_loader(loader):
    epoch = 0
    while True:
        loader.sampler.set_epoch(epoch)
        for batch in loader:
            yield batch
        epoch += 1
