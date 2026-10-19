"""Explicit, checkpointed experiment matrix."""
from dataclasses import dataclass
from .reference_config import TrainConfig

VARIANTS = ('baseline', 'block_skew', 'dense_circle', 'block_skew_circle')
DATASETS = ('unblocked', 'blocked_per_source')

@dataclass
class Config(TrainConfig):
    variant: str = 'baseline'
    block_size: int = 7
    protocol: str = 'lie_ssl_k1_v1'

    def __post_init__(self):
        super().__post_init__()
        if self.variant not in VARIANTS:
            raise ValueError(f'variant must be one of {VARIANTS}')
        if self.latent_dim % 2:
            raise ValueError('This comparison requires an even latent dimension')
        if self.protocol != 'lie_ssl_k1_v1':
            raise ValueError('Incompatible protocol')

    @property
    def circle(self):
        return self.variant.endswith('circle')

    @property
    def structured(self):
        return self.variant.startswith('block_skew')

    @property
    def resolved_run_name(self):
        return self.run_name or f'{self.dataset_variant}__{self.variant}__s{self.seed}'

    def to_dict(self):
        result = super().to_dict()
        result.update(generator_structure='block_skew_2x2' if self.structured else 'dense',
                      coefficient_parameterization='circle' if self.circle else 'scalar',
                      generator_normalization='unit_block_frequency' if self.structured else
                          ('sqrt_d_frobenius' if self.circle else 'unit_frobenius'),
                      K=1)
        return result


def from_dict(raw):
    return Config(**{k:v for k,v in raw.items() if k in Config.__dataclass_fields__})


def setting(index, data_root, output_root, smoke=False):
    if not 0 <= index < 8:
        raise ValueError('Setting index must be 0..7')
    return Config(data_root=str(data_root), output_root=str(output_root),
                  dataset_variant=DATASETS[index//4], variant=VARIANTS[index%4],
                  epochs=2 if smoke else 300, batch_size=32 if smoke else 256,
                  num_workers=0 if smoke else 4,
                  max_train_batches=2 if smoke else 0,
                  max_val_batches=2 if smoke else 0,
                  max_test_batches=2 if smoke else 0)
