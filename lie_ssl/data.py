"""Matched unblocked and per-source blocked rotated-MNIST datasets.

Both variants retain the original 28x28 canvas.  The blocked variant rotates first and
then applies one source-deterministic square in image coordinates.  A local generator
keeps block placement from perturbing the rotation or DataLoader RNG streams.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .reference_config import TrainConfig
from .utils import seed_worker


def rotate_image(image: torch.Tensor, angle_degrees: float) -> torch.Tensor:
    return TF.rotate(
        image,
        angle=float(angle_degrees),
        interpolation=InterpolationMode.BILINEAR,
        expand=False,
        fill=0.0,
    )


def stratified_split_indices(
    targets: torch.Tensor, val_fraction: float = 0.1, split_seed: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    indices = np.arange(len(targets), dtype=np.int64)
    train_indices, val_indices = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=split_seed,
        shuffle=True,
        stratify=targets.cpu().numpy(),
    )
    return torch.from_numpy(train_indices), torch.from_numpy(val_indices)


class BlockedMNIST(Dataset):
    """MNIST views for the matched unblocked/blocked-per-source comparison."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        *,
        two_view: bool = True,
        val_fraction: float = 0.1,
        split_seed: int = 42,
        fixed_seed: Optional[int] = None,
        block_size: int = 10,
        block_fill: float = 1.0,
        block_margin: int = 1,
        block_seed: int = 1234,
        dataset_variant: str = "blocked_per_source",
        download: bool = False,
    ) -> None:
        super().__init__()
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be train, val, or test")
        if dataset_variant not in {"unblocked", "blocked_per_source"}:
            raise ValueError(
                "dataset_variant must be unblocked or blocked_per_source"
            )
        self.split = split
        self.two_view = bool(two_view)
        self.fixed_seed = fixed_seed
        self.random_views = fixed_seed is None
        self.block_size = int(block_size)
        self.block_fill = float(block_fill)
        self.block_margin = int(block_margin)
        self.block_seed = int(block_seed)
        self.dataset_variant = dataset_variant

        self.image_size = 28
        span = self.image_size - self.block_size - 2 * self.block_margin
        if self.block_size <= 0 or span < 0:
            raise ValueError("invalid block size or margin for the 28x28 canvas")

        base_train = split in {"train", "val"}
        mnist = MNIST(root=str(root), train=base_train, download=download)
        if base_train:
            train_indices, val_indices = stratified_split_indices(
                mnist.targets, val_fraction=val_fraction, split_seed=split_seed
            )
            indices = train_indices if split == "train" else val_indices
        else:
            indices = torch.arange(len(mnist.data), dtype=torch.long)

        self.images = (mnist.data[indices].to(torch.float32) / 255.0).contiguous()
        self.labels = mnist.targets[indices].to(torch.long).contiguous()
        self.source_indices = indices.to(torch.long).contiguous()

    def __len__(self) -> int:
        return int(self.images.shape[0])

    def base_image(self, index: int) -> torch.Tensor:
        """Return the unoccluded canonical 28x28 canvas."""
        return self.images[index].unsqueeze(0)

    def _draw_source_blocks(self, index: int) -> torch.Tensor:
        span = self.image_size - self.block_size - 2 * self.block_margin
        source_index = int(self.source_indices[index])
        generator = torch.Generator().manual_seed(
            self.block_seed * 1_000_003 + source_index
        )
        draw = torch.randint(0, span + 1, (1, 2), generator=generator)
        return draw + self.block_margin

    def sample_context(self, index: int) -> torch.Tensor:
        if self.dataset_variant == "unblocked":
            return torch.empty((0, 2), dtype=torch.long)
        return self._draw_source_blocks(index)

    def apply_blocks(self, image: torch.Tensor, corners: torch.Tensor) -> torch.Tensor:
        output = image.clone()
        for top, left in corners.tolist():
            output[
                ..., top : top + self.block_size, left : left + self.block_size
            ] = self.block_fill
        return output

    def render(
        self, index: int, angle_degrees: float, corners: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        context = self.sample_context(index) if corners is None else corners
        rotated = rotate_image(self.base_image(index), angle_degrees)
        return self.apply_blocks(rotated, context)

    def occluded_base_image(
        self, index: int, corners: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        context = self.sample_context(index) if corners is None else corners
        return self.apply_blocks(self.base_image(index), context)

    def orbit(
        self, index: int, angles_degrees: torch.Tensor | list[float]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.sample_context(index)
        images = torch.stack(
            [self.render(index, float(angle), context) for angle in angles_degrees]
        )
        return images, context.clone()

    def _angles(self, index: int) -> tuple[float, Optional[float]]:
        if self.random_views:
            first = 360.0 * random.random()
            second = 360.0 * random.random() if self.two_view else None
            return first, second
        generator = random.Random(int(self.fixed_seed) + int(index))
        first = 360.0 * generator.random()
        second = 360.0 * generator.random() if self.two_view else None
        return first, second

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        first_angle, second_angle = self._angles(index)
        context = self.sample_context(index)
        item = {
            "x_i": self.render(index, first_angle, context),
            "angle_i": torch.tensor(first_angle, dtype=torch.float32),
            "label": self.labels[index].clone(),
            "source_index": self.source_indices[index].clone(),
            "dataset_index": torch.tensor(index, dtype=torch.long),
            "block_corners": context.clone(),
        }
        if self.two_view:
            assert second_angle is not None
            item["x_j"] = self.render(index, second_angle, context)
            item["angle_j"] = torch.tensor(second_angle, dtype=torch.float32)
        return item


@dataclass
class LoaderBundle:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    train_dataset: BlockedMNIST
    val_dataset: BlockedMNIST
    test_dataset: BlockedMNIST
    generators: dict[str, torch.Generator]


def _dataset_kwargs(config: TrainConfig) -> dict:
    return {
        "root": config.data_root,
        "val_fraction": config.val_fraction,
        "split_seed": config.split_seed,
        "block_size": config.block_size,
        "block_fill": config.block_fill,
        "block_margin": config.block_margin,
        "block_seed": config.block_seed,
        "dataset_variant": config.dataset_variant,
        "download": False,
    }


def build_datasets(
    config: TrainConfig, *, two_view: bool = True
) -> tuple[BlockedMNIST, BlockedMNIST, BlockedMNIST]:
    common = _dataset_kwargs(config)
    train = BlockedMNIST(
        split="train", two_view=two_view, fixed_seed=None, **common
    )
    val = BlockedMNIST(
        split="val",
        two_view=two_view,
        fixed_seed=config.val_rotation_seed,
        **common,
    )
    test = BlockedMNIST(
        split="test",
        two_view=two_view,
        fixed_seed=config.test_rotation_seed,
        **common,
    )
    return train, val, test


def build_loaders(config: TrainConfig, *, two_view: bool = True) -> LoaderBundle:
    train_dataset, val_dataset, test_dataset = build_datasets(
        config, two_view=two_view
    )
    generators = {
        "train": torch.Generator().manual_seed(config.seed),
        "val": torch.Generator().manual_seed(config.seed + 1),
        "test": torch.Generator().manual_seed(config.seed + 2),
    }
    common = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
        "worker_init_fn": seed_worker,
    }
    train = DataLoader(
        train_dataset, shuffle=True, generator=generators["train"], **common
    )
    val = DataLoader(
        val_dataset, shuffle=False, generator=generators["val"], **common
    )
    test = DataLoader(
        test_dataset, shuffle=False, generator=generators["test"], **common
    )
    return LoaderBundle(
        train=train,
        val=val,
        test=test,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        generators=generators,
    )
