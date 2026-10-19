"""Deterministic loading, provenance, atomic checkpoints, and offline logging."""
import hashlib
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import build_datasets
from .utils import seed_everything, seed_worker, limited_batches, write_json, write_history_csv
from .config import from_dict
from .model import GroupLatentAutoencoder


def device():
    torch.set_num_threads(min(8, int(os.environ.get('SLURM_CPUS_PER_TASK', '8'))))
    if not torch.cuda.is_available() or not torch.version.hip:
        raise RuntimeError('GPU jobs require a ROCm-enabled PyTorch on LUMI-G')
    return torch.device('cuda')


def epoch_loader(dataset, config, epoch=0, training=False, seed=None):
    value = (config.seed if seed is None else seed) + max(epoch-1, 0)
    if training:
        seed_everything(value)
    return DataLoader(dataset, batch_size=config.batch_size, shuffle=training,
                      num_workers=config.num_workers, pin_memory=True,
                      worker_init_fn=seed_worker, persistent_workers=False,
                      generator=torch.Generator().manual_seed(value))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def data_provenance(config, datasets):
    raw = Path(config.data_root)/'MNIST/raw'
    names = ('train-images-idx3-ubyte', 'train-labels-idx1-ubyte',
             't10k-images-idx3-ubyte', 't10k-labels-idx1-ubyte')
    return dict(raw_sha256={n:sha256(raw/n) for n in names},
                splits={d.split:dict(samples=len(d), source_sha256=hashlib.sha256(
                    d.source_indices.numpy().tobytes()).hexdigest()) for d in datasets},
                contract={k:getattr(config,k) for k in ('dataset_variant','block_size','block_fill',
                    'block_margin','block_seed','split_seed','val_fraction',
                    'val_rotation_seed','test_rotation_seed')})


def save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                gpu=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['gpu']:
        torch.cuda.set_rng_state_all(state['gpu'])


def compatible(config, checkpoint, data):
    old, new = checkpoint['config'], config.to_dict()
    ignored = {'epochs','output_root','data_root','wandb_mode','wandb_project','wandb_group'}
    mismatch = [k for k in new if k not in ignored and old.get(k) != new[k]]
    if mismatch or checkpoint['data'] != data or checkpoint['epoch'] > config.epochs:
        raise ValueError(f'Incompatible resume: {mismatch} or changed dataset/epoch budget')


def frozen_model(checkpoint, target_device):
    payload = load(checkpoint)
    config = from_dict(payload['config'])
    model = GroupLatentAutoencoder(config).to(target_device)
    model.load_state_dict(payload['model'], strict=True)
    model.eval().requires_grad_(False)
    return model, config, payload


def state_copy(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def assert_frozen(model, before):
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise AssertionError('Backbone is not frozen')
    if any(not torch.equal(v.detach().cpu(), before[k]) for k,v in model.state_dict().items()):
        raise AssertionError('Backbone state changed')


class Logger:
    def __init__(self, directory, name, config, mode='offline'):
        self.run = None
        if mode != 'disabled':
            import wandb
            self.run = wandb.init(project='lie_SSL', group=os.environ.get('LIE_GROUP','k1_mnist'),
                                  name=name, config=config, dir=str(directory), mode=mode,
                                  settings=wandb.Settings(init_timeout=120))
    def log(self, values):
        if self.run:
            self.run.log(values)
    def close(self):
        if self.run:
            self.run.finish()
