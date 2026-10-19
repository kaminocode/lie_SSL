"""Reference encoder/decoder with explicit generator and coefficient ablations."""
import math
import torch
from torch import nn
from .reference_model import ConvEncoder, ConvDecoder, decode_with_action


def circle_coordinates(raw):
    # Nonzero initialization avoids the singularity; the fallback is defined at zero.
    norm = torch.linalg.vector_norm(raw.float(), dim=-1, keepdim=True)
    unit = raw.float() / norm.clamp_min(1e-8)
    fallback = torch.zeros_like(unit)
    fallback[..., 0] = 1
    return torch.where(norm > 1e-8, unit, fallback)


def principal_angle(unit):
    angle = torch.atan2(unit[..., 1:2], unit[..., 0:1])
    return torch.where(angle <= -math.pi, angle + 2*math.pi, angle)


class Encoder(ConvEncoder):
    def __init__(self, dim, circle):
        super().__init__(dim)
        self.circle = circle
        if circle:
            self.pose_head = nn.Linear(256, 2)
            nn.init.zeros_(self.pose_head.weight)
            with torch.no_grad():
                self.pose_head.bias.copy_(torch.tensor([1., 0.]))


class LieAction(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.structured, self.circle = config.structured, config.circle
        self.target_norm = math.sqrt(config.latent_dim) if config.circle else 1.
        if self.structured:
            block = torch.tensor([[0., -1.], [1., 0.]])
            self.register_buffer('B', torch.block_diag(*[block for _ in range(config.latent_dim//2)]))
        else:
            self.B = nn.Parameter(.1 * torch.randn(config.latent_dim, config.latent_dim))
        self.normalize_generator_()

    def matrix(self, raw_pose):
        coefficient = principal_angle(circle_coordinates(raw_pose)) if self.circle else raw_pose.float()
        if self.structured:
            identity = torch.eye(self.B.shape[0], device=self.B.device)
            action = coefficient.cos().unsqueeze(-1)*identity + coefficient.sin().unsqueeze(-1)*self.B
        else:
            action = torch.matrix_exp(coefficient.unsqueeze(-1)*self.B.float().unsqueeze(0))
        return action, coefficient

    def forward(self, latent, raw_pose):
        action, coefficient = self.matrix(raw_pose)
        posed = torch.bmm(action, latent.float().unsqueeze(-1)).squeeze(-1)
        return posed.to(latent.dtype), action, coefficient

    @torch.no_grad()
    def normalize_generator_(self):
        if not self.structured:
            norm = torch.linalg.matrix_norm(self.B, ord='fro')
            if not torch.isfinite(norm) or norm < 1e-12:
                raise FloatingPointError('Cannot normalize a zero or nonfinite generator')
            self.B.mul_(self.target_norm/norm)


class GroupLatentAutoencoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.latent_dim, self.circle = config.latent_dim, config.circle
        self.encoder = Encoder(config.latent_dim, config.circle)
        self.action = LieAction(config)
        self.decoder = ConvDecoder(config.latent_dim)

    def forward(self, image):
        canonical, raw_pose = self.encoder(image)
        posed, action, coefficient = self.action(canonical, raw_pose)
        return dict(z=canonical, raw_pose=raw_pose, pose_coefficient=coefficient,
                    B=self.action.B, A=action, z_posed=posed,
                    observed_recon=self.decoder(posed))

    def representation(self, image, name):
        canonical, pose = self.encoder(image)
        if name == 'canonical':
            return canonical
        if name != 'pose':
            raise ValueError(name)
        return circle_coordinates(pose) if self.circle else pose

    def normalize_generator_(self):
        self.action.normalize_generator_()
