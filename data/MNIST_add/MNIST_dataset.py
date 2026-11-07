"""Multi-digit MNIST addition (adapted from A-NeSI / DeepProbLog). MNIST train is split 50k/10k into train/val."""
import random
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset as TorchDataset
from torch.utils.data import random_split


_DATA_ROOT = Path(__file__).parent

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

_full_train_set = torchvision.datasets.MNIST(
    root=str(_DATA_ROOT), train=True, download=True, transform=transform
)
_train_set, _val_set = random_split(_full_train_set, [50000, 10000], generator=torch.Generator().manual_seed(0))

datasets = {
    "train": _train_set,
    "val": _val_set,
    "full_train": _full_train_set,
    "test": torchvision.datasets.MNIST(
        root=str(_DATA_ROOT), train=False, download=True, transform=transform
    ),
}


def digits_to_number(digits: Iterable[int]) -> int:
    number = 0
    for d in digits:
        number *= 10
        number += d
    return number


def number_to_digits(n: int) -> List[int]:
    return [int(c) for c in str(n)]


def addition(n: int, dataset: str, seed=None, output_digits=False):
    """Sums of two n-digit numbers. dataset: 'train', 'val', 'test' or 'full_train'."""
    return MNISTOperator(
        dataset_name=dataset,
        operator=sum,
        size=n,
        arity=2,
        seed=seed,
        output_digits=output_digits,
    )


class MNISTOperator(TorchDataset):
    """Examples of `arity` numbers with `size` digits; each image used once. Item: (images 1, images 2,
    label, digit labels), digits most significant first. With output_digits the label is the list of
    sum digits, least significant first."""
    def __init__(
        self,
        dataset_name: str,
        operator: Callable[[List[int]], int],
        size=1,
        arity=2,
        seed=None,
        output_digits=False,
    ):
        super(MNISTOperator, self).__init__()
        assert size >= 1
        assert arity >= 1
        self.dataset_name = dataset_name
        self.dataset = datasets[self.dataset_name]
        self.operator = operator
        self.size = size
        self.arity = arity
        self.seed = seed
        self.output_digits = output_digits

        mnist_indices = list(range(len(self.dataset)))
        if seed is not None:
            random.Random(seed).shuffle(mnist_indices)
        dataset_iter = iter(mnist_indices)
        # every example: `arity` lists of `size` mnist indices; stops when the images run out
        self.data = []
        try:
            while True:
                self.data.append([[next(dataset_iter) for _ in range(self.size)] for _ in range(self.arity)])
        except StopIteration:
            pass

    def __getitem__(self, index: int) -> Tuple[np.array, np.array, int, List[int]]:
        l1, l2 = self.data[index]
        label, digits = self._get_label(index)
        l1 = torch.stack([self.dataset[x][0][0] for x in l1])
        l2 = torch.stack([self.dataset[x][0][0] for x in l2])
        return l1, l2, label, digits

    def _get_label(self, i: int):
        digits = [[self.dataset[j][1] for j in number] for number in self.data[i]]
        result = self.operator([digits_to_number(d) for d in digits])
        if not self.output_digits:
            return result, digits
        result_digits = number_to_digits(result)
        result_digits = [0] * (self.size + 1 - len(result_digits)) + result_digits  # pad to size+1 digits
        return result_digits[::-1], digits  # least significant digit first

    def __len__(self):
        return len(self.data)
