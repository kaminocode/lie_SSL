"""Configuration contract shared by training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class TrainConfig:
    data_root: str
    output_root: str
    latent_dim: int = 24
    dataset_variant: str = "blocked_per_source"
    block_size: int = 10
    block_fill: float = 1.0
    block_margin: int = 1
    block_seed: int = 1234
    seed: int = 42
    split_seed: int = 42
    val_fraction: float = 0.1
    val_rotation_seed: int = 0
    test_rotation_seed: int = 1_000_000
    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 300
    lr: float = 1e-3
    weight_decay: float = 0.0
    lambda_obs: float = 1.0
    lambda_swap: float = 1.0
    lambda_can: float = 1.0
    max_train_batches: int = 0
    max_val_batches: int = 0
    max_test_batches: int = 0
    wandb_project: str = "unsupervised_g"
    wandb_group: str = "experiment_1_rotated_mnist_blocked_28x28"
    wandb_mode: str = "offline"
    setting_id: str = "baseline"
    run_name: str = ""

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if self.dataset_variant not in {"unblocked", "blocked_per_source"}:
            raise ValueError(
                "dataset_variant must be unblocked or blocked_per_source"
            )
        if self.block_size <= 0:
            raise ValueError("block_size must be positive")
        if self.block_size + 2 * self.block_margin > self.image_size:
            raise ValueError("the requested block and margin do not fit the canvas")
        if not 0.0 < self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in (0, 1)")
        if self.batch_size <= 0 or self.num_workers < 0 or self.epochs <= 0:
            raise ValueError("invalid batch_size, num_workers, or epochs")
        if self.lr <= 0 or self.weight_decay < 0:
            raise ValueError("invalid optimizer settings")
        coefficients = (self.lambda_obs, self.lambda_swap, self.lambda_can)
        if any(value < 0 for value in coefficients):
            raise ValueError("loss coefficients must be non-negative")
        if not any(value > 0 for value in coefficients):
            raise ValueError("at least one loss coefficient must be positive")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled")
        if not self.setting_id:
            raise ValueError("setting_id cannot be empty")
        for name in ("max_train_batches", "max_val_batches", "max_test_batches"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")

    @property
    def image_size(self) -> int:
        return 28

    @property
    def resolved_run_name(self) -> str:
        return self.run_name or (
            f"rotated_lie_d{self.latent_dim}_{self.dataset_variant}_28x28_s{self.seed}"
        )

    @property
    def run_dir(self) -> Path:
        return Path(self.output_root).expanduser().resolve() / self.resolved_run_name

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["image_size"] = self.image_size
        payload["run_name"] = self.resolved_run_name
        return payload


def config_from_checkpoint(payload: dict[str, Any]) -> TrainConfig:
    """Restore the public training configuration from a checkpoint payload."""
    raw = dict(payload["config"])
    raw.pop("image_size", None)
    allowed = TrainConfig.__dataclass_fields__
    return TrainConfig(**{key: value for key, value in raw.items() if key in allowed})
