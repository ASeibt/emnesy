"""Dataloaders for Warcraft shortest path: data_root/<k>x<k>/{train,val,test}_{maps,shortest_paths,vertex_weights}.npy
(train split into _part0/_part1 for k > 12)."""
from __future__ import annotations

import einops as E
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

DEFAULT_DATA_ROOT = "data/warcraft/warcraft_shortest_path_oneskin"


def load_warcraft_from_disk(grid_size: int, split: str, type: str, data_root: str = DEFAULT_DATA_ROOT) -> np.ndarray:
    """type: 'maps', 'shortest_paths' or 'vertex_weights'."""
    assert split in ["train", "val", "test"]

    data_path = f"{data_root}/{grid_size}x{grid_size}"

    if grid_size > 12 and split == "train":
        split_data1 = np.load(f"{data_path}/{split}_{type}_part0.npy").astype(
            np.float32
        )
        split_data2 = np.load(f"{data_path}/{split}_{type}_part1.npy").astype(
            np.float32
        )
        split_data = np.concatenate((split_data1, split_data2), axis=0)
    else:
        split_data = np.load(f"{data_path}/{split}_{type}.npy").astype(np.float32)

    if not np.all(np.isfinite(split_data)):
        raise ValueError(f"Data loaded from {data_path} contains non-finite values.")

    return split_data


def normalise(data: np.ndarray) -> np.ndarray:
    """Normalise the map pixels."""
    mean, std = (
        np.mean(data, axis=(0, 2, 3), keepdims=True),
        np.std(data, axis=(0, 2, 3), keepdims=True),
    )
    # Add epsilon to std to avoid division by zero
    std = np.maximum(std, 1e-8)
    return (data - mean) / std


def make_dataloader(
    grid_size: int,
    batch_size: int,
    maps: np.ndarray,
    target: np.ndarray,
    *,
    shuffle: bool = True,
    generator: torch.Generator | None = None,
) -> DataLoader:
    """Maps as 8x8 tiles per cell, target (B, N, 2) = (path, cell cost)."""
    maps_tensor = torch.tensor(maps, dtype=torch.float32)
    # re-arrange into tiles of 8x8 pixels per cell, as expected by PartialResNet
    maps_tensor = E.rearrange(
        maps_tensor, "b (g1 w) (g2 h) c -> b g1 g2 w h c", g1=grid_size, g2=grid_size
    )

    target_tensor = torch.tensor(target, dtype=torch.float32)

    if not torch.all(torch.isfinite(maps_tensor)):
        raise ValueError("Input maps contain non-finite values.")
    if not torch.all(torch.isfinite(target_tensor)):
        raise ValueError("Input targets contain non-finite values.")

    dataset = TensorDataset(maps_tensor, target_tensor)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=True,
        generator=generator,
    )


def load_warcraft_data(
    grid_size: int,
    batch_size: int,
    split: str,
    *,
    generator: torch.Generator | None = None,
    data_root: str = DEFAULT_DATA_ROOT,
) -> DataLoader:
    """Loader for one split."""
    if grid_size <= 0:
        raise ValueError(f"grid_size must be positive, got {grid_size}")

    # load maps and normalise pixel values
    maps = load_warcraft_from_disk(grid_size, split, "maps", data_root)
    maps = normalise(maps)

    # load shortest paths (labels: 0 or 1)
    labels = load_warcraft_from_disk(grid_size, split, "shortest_paths", data_root)

    # load the cell costs (used for the accuracy of the predicted costs)
    weights = load_warcraft_from_disk(grid_size, split, "vertex_weights", data_root)

    # target will be (B, N, 2) where [..., 0] is path label and [..., 1] is weights
    target = np.stack([labels, weights], axis=-1)

    # Flatten grid for target
    target = E.rearrange(target, "b g1 g2 o -> b (g1 g2) o")

    shuffle = split == "train"
    data = make_dataloader(
        grid_size, batch_size, maps, target, shuffle=shuffle, generator=generator
    )
    return data


def create_loader(
    grid_size: int, batch_size: int = 10, *, seed: int | None = None, data_root: str = DEFAULT_DATA_ROOT, **kwargs
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train, val and test loaders; incomplete batches are dropped."""
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    if "val_batch_size" in kwargs:
        val_batch_size = kwargs["val_batch_size"]
    else:
        val_batch_size = batch_size

    if "test_batch_size" in kwargs:
        test_batch_size = kwargs["test_batch_size"]
    else:
        test_batch_size = batch_size

    train_loader = load_warcraft_data(
        grid_size, batch_size, "train", generator=generator, data_root=data_root
    )
    val_loader = load_warcraft_data(
        grid_size, val_batch_size, "val", generator=generator, data_root=data_root
    )
    test_loader = load_warcraft_data(
        grid_size, test_batch_size, "test", generator=generator, data_root=data_root
    )
    return train_loader, val_loader, test_loader