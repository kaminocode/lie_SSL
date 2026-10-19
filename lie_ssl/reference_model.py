"""The Experiment 1 Lie autoencoder on its original 28x28 canvas."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
        )
        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 256),
            nn.ELU(),
        )
        self.canonical_head = nn.Linear(256, latent_dim)
        self.pose_head = nn.Linear(256, 1)
        nn.init.zeros_(self.pose_head.weight)
        nn.init.zeros_(self.pose_head.bias)

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(self.features(image))
        return self.canonical_head(features), self.pose_head(features)


class ConvDecoder(nn.Module):
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__()
        self.fully_connected = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, 64 * 2 * 2),
            nn.ELU(),
            nn.Unflatten(dim=1, unflattened_size=(64, 2, 2)),
        )
        self.upsampling = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1),  # 2 -> 4
            nn.ELU(),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=0),  # 4 -> 7
            nn.ELU(),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),  # 7 -> 14
            nn.ELU(),
            nn.ConvTranspose2d(32, 1, 3, 2, 1, output_padding=1),  # 14 -> 28
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.upsampling(self.fully_connected(latent))


class LieAction(nn.Module):
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__()
        self.B = nn.Parameter(0.1 * torch.randn(latent_dim, latent_dim))

    def matrix(self, raw_pose: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficient = raw_pose.float()
        lie_element = coefficient.unsqueeze(-1) * self.B.float().unsqueeze(0)
        return torch.matrix_exp(lie_element), coefficient

    def forward(
        self, latent: torch.Tensor, raw_pose: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action, coefficient = self.matrix(raw_pose)
        posed = torch.bmm(action, latent.float().unsqueeze(-1)).squeeze(-1)
        return posed.to(latent.dtype), action, coefficient

    @torch.no_grad()
    def normalize_generator_(self) -> None:
        norm = torch.linalg.matrix_norm(self.B, ord="fro")
        self.B.div_(norm.clamp_min(1e-8))


class GroupLatentAutoencoder(nn.Module):
    def __init__(self, latent_dim: int = 24) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.encoder = ConvEncoder(latent_dim)
        self.action = LieAction(latent_dim)
        self.decoder = ConvDecoder(latent_dim)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        canonical, raw_pose = self.encoder(image)
        posed, action, coefficient = self.action(canonical, raw_pose)
        observed = self.decoder(posed)
        return {
            "z": canonical,
            "raw_pose": raw_pose,
            "pose_coefficient": coefficient,
            "B": self.action.B,
            "A": action,
            "z_posed": posed,
            "observed_recon": observed,
        }

    @torch.no_grad()
    def normalize_generator_(self) -> None:
        self.action.normalize_generator_()


def decode_with_action(
    model: GroupLatentAutoencoder, latent: torch.Tensor, action: torch.Tensor
) -> torch.Tensor:
    acted = torch.bmm(action.float(), latent.float().unsqueeze(-1)).squeeze(-1)
    return model.decoder(acted.to(latent.dtype))
