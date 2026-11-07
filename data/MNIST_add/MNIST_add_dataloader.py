"""Dataloaders for MNIST addition."""
import torch
from torch.utils.data import DataLoader

from data.MNIST_add.MNIST_dataset import addition


def get_mnist_addition_dataloaders(
    n_digits: int = 1,
    batch_size: int = 32,
    shuffle_train: bool = True,
    num_workers: int = 0,
    seed: int = 0,
    output_digits: bool = False,
):
    """Train, val and test loaders. Batch with output_digits: (images 1, images 2, sum digits
    (B, n_digits + 1) least significant first, digit labels). Incomplete batches are dropped."""
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        torch.manual_seed(worker_seed)
        torch.cuda.manual_seed(worker_seed)

    def addition_collate_fn(batch):
        l1, l2, labels, digits = zip(*batch)
        return torch.stack(l1), torch.stack(l2), torch.tensor(labels, dtype=torch.long), digits

    collate = addition_collate_fn if output_digits else None

    def make_loader(split, shuffle):
        return DataLoader(
            addition(n=n_digits, dataset=split, seed=seed, output_digits=output_digits),
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=torch.cuda.is_available(),
            collate_fn=collate,
            drop_last=True,
        )

    train_loader = make_loader("train", shuffle_train)
    val_loader = make_loader("val", False)
    test_loader = make_loader("test", False)
    return train_loader, val_loader, test_loader
