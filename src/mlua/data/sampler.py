import itertools
from typing import List, Iterator, Tuple

import numpy as np
from torch.utils.data.sampler import Sampler


def iterate_once(iterable: List[int]) -> np.ndarray:
    return np.random.permutation(iterable)


def iterate_eternally(indices: List[int]) -> Iterator[int]:
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable: Iterator[int], n: int) -> Iterator[Tuple[int, ...]]:
    """Collect data into fixed-length chunks or blocks."""
    args = [iter(iterable)] * n
    return zip(*args)


class TwoStreamBatchSampler(Sampler):
    """
    Official MLUA TwoStreamBatchSampler for semi-supervised training.
    Iterates through labeled indices once per epoch while continuously drawing
    unlabeled indices with infinite permutation streaming.
    """
    def __init__(
        self,
        l_indices: List[int],
        ul_indices: List[int],
        batch_size: int = 8,
        l_batch_size: int = 4,
    ):
        self.l_indices = list(l_indices)
        self.ul_indices = list(ul_indices)
        self.batch_size = batch_size
        self.l_batch_size = l_batch_size
        self.ul_batch_size = batch_size - l_batch_size

        assert len(self.l_indices) >= self.l_batch_size > 0, (
            f"Labeled samples ({len(self.l_indices)}) must be >= labeled batch size ({self.l_batch_size})"
        )
        assert len(self.ul_indices) >= self.ul_batch_size >= 0, (
            f"Unlabeled samples ({len(self.ul_indices)}) must be >= unlabeled batch size ({self.ul_batch_size})"
        )

    def __iter__(self) -> Iterator[List[int]]:
        label_iter = iterate_once(self.l_indices)
        unlabel_iter = iterate_eternally(self.ul_indices)

        if self.ul_batch_size == 0:
            return (
                list(l_batch + l_batch)
                for (l_batch, l_batch) in zip(
                    grouper(label_iter, self.l_batch_size),
                    grouper(label_iter, self.l_batch_size),
                )
            )

        return (
            list(l_batch + ul_batch)
            for (l_batch, ul_batch) in zip(
                grouper(label_iter, self.l_batch_size),
                grouper(unlabel_iter, self.ul_batch_size),
            )
        )

    def __len__(self) -> int:
        return len(self.l_indices) // self.l_batch_size
