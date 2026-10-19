"""The three-term Experiment 1 objective."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import GroupLatentAutoencoder, decode_with_action


def compute_method_losses(
    model: GroupLatentAutoencoder,
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    lambda_obs: float = 1.0,
    lambda_swap: float = 1.0,
    lambda_can: float = 1.0,
) -> dict[str, torch.Tensor]:
    first_output = model(first)
    second_output = model(second)

    observed = 0.5 * (
        F.mse_loss(first_output["observed_recon"], first)
        + F.mse_loss(second_output["observed_recon"], second)
    )
    swapped_second = decode_with_action(
        model, first_output["z"], second_output["A"]
    )
    swapped_first = decode_with_action(
        model, second_output["z"], first_output["A"]
    )
    swapped = 0.5 * (
        F.mse_loss(swapped_second, second) + F.mse_loss(swapped_first, first)
    )
    canonical = F.mse_loss(first_output["z"].float(), second_output["z"].float())
    common_total = observed + swapped + canonical
    total = lambda_obs * observed + lambda_swap * swapped + lambda_can * canonical
    return {
        "total": total,
        "common_total": common_total,
        "obs": observed,
        "swap": swapped,
        "can": canonical,
    }
