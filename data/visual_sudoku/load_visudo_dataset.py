"""Dataloaders for ViSudo-PC (Sudoku grids of MNIST images, labelled valid / invalid).

Expected layout: <data_root>/dimension__<grid_size>/datasets__mnist/strategy__simple/strategy__simple/
numTrain__<num_train>/numTest__00100/numValid__00100/corruptChance__0.50/overlap__<overlap>/split__<split>/"""
import einops as E
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

MNIST_DIM = 28
DEFAULT_DATA_ROOT = "data/visual_sudoku/visudo"


def load_visudo(
    grid_size: int,
    partition: str,
    num_train: str,
    overlap: str,
    split: int,
    use_negative: bool,
    data_root: str = DEFAULT_DATA_ROOT,
):
    """Grids (N, grid_size, grid_size, 28, 28) and labels (N,) of one partition; without use_negative
    only valid Sudokus."""
    data_dir = (
        data_root
        + f"/dimension__{grid_size}"
        + "/datasets__mnist/strategy__simple"
        + "/strategy__simple"
        + f"/numTrain__{num_train}"
        + "/numTest__00100"
        + "/numValid__00100"
        + "/corruptChance__0.50"
        + f"/overlap__{overlap}"
        + f"/split__{split:02d}"
    )

    labels = np.loadtxt(data_dir + f"/{partition}_puzzle_labels.txt", delimiter="\t", dtype=int)[:, 0]
    grids = np.loadtxt(data_dir + f"/{partition}_puzzle_pixels.txt", delimiter="\t", dtype=float)

    if not use_negative:
        indices = np.where(labels == 1)[0]
        grids = grids[indices]
        labels = labels[indices]

    grids = E.rearrange(
        grids,
        "b (gs1 gs2 md1 md2) -> b gs1 gs2 md1 md2",
        gs1=grid_size,
        gs2=grid_size,
        md1=MNIST_DIM,
        md2=MNIST_DIM,
    )
    return grids, labels


def create_loader(
    grid_size: int,
    batch_size: int = 10,
    num_train: str = "00100",
    overlap: str = "0.00",
    split: int = 11,
    use_negative: bool = False,
    num_workers: int = 0,
    seed: int = 0,
    data_root: str = DEFAULT_DATA_ROOT,
):
    """Train, val and test loaders. use_negative only affects train. Incomplete batches are dropped."""
    torch_gen = torch.Generator().manual_seed(seed)

    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def make_loader(data, shuffle):
        dataset = TensorDataset(
            torch.tensor(data[0], dtype=torch.get_default_dtype()),
            torch.tensor(data[1], dtype=torch.get_default_dtype()),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            generator=torch_gen,
            worker_init_fn=seed_worker,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    train_data = load_visudo(grid_size, "train", num_train, overlap, split, use_negative, data_root)
    val_data = load_visudo(grid_size, "valid", num_train, overlap, split, True, data_root)
    test_data = load_visudo(grid_size, "test", num_train, overlap, split, True, data_root)
    return make_loader(train_data, shuffle=True), make_loader(val_data, shuffle=False), make_loader(test_data, shuffle=False)


if __name__ == "__main__":
    # quick check of the data: python -m data.visual_sudoku.load_visudo_dataset
    train_loader, val_loader, test_loader = create_loader(grid_size=4, batch_size=10, split=11)
    data, target = next(iter(val_loader))
    print(f"data shape: {data.shape}")
    print(f"target shape: {target.shape}")  # (batch_size,)
    print("train/val/test size:", len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))
