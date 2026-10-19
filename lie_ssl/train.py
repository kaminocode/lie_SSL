"""Resumable backbone training with the unchanged three-term objective."""
import time
import numpy as np
import torch
from .model import GroupLatentAutoencoder
from .losses import compute_method_losses
from .runtime import (build_datasets, device, seed_everything, epoch_loader, data_provenance,
    limited_batches, load, save, rng_state, restore_rng, compatible, Logger, write_json, write_history_csv)


def epoch(model, config, loader, target_device, optimizer=None, maximum=0):
    model.train(optimizer is not None)
    sums, samples = {}, 0
    for batch in limited_batches(loader, maximum):
        first, second = (batch[k].to(target_device, non_blocking=True) for k in ('x_i','x_j'))
        with torch.set_grad_enabled(optimizer is not None):
            loss = compute_method_losses(model, first, second, lambda_obs=config.lambda_obs,
                                         lambda_swap=config.lambda_swap, lambda_can=config.lambda_can)
        if not all(torch.isfinite(v) for v in loss.values()):
            raise FloatingPointError('Nonfinite backbone loss')
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss['total'].backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('Nonfinite backbone gradients')
            optimizer.step()
            model.normalize_generator_()
        samples += len(first)
        for name, value in loss.items():
            sums[name] = sums.get(name, 0.) + float(value.detach())*len(first)
    if not samples:
        raise ValueError('Empty training/evaluation split')
    return {k:v/samples for k,v in sums.items()}


def train(config, target_device=None, stop_after=None):
    target_device = target_device or device()
    seed_everything(config.seed)
    datasets = build_datasets(config)
    provenance = data_provenance(config, datasets)
    model = GroupLatentAutoencoder(config).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    directory = config.run_dir
    directory.mkdir(parents=True, exist_ok=True)
    last = directory/'last.pt'
    start, best, best_epoch, history = 1, float('inf'), 0, []
    if last.exists():
        previous = load(last)
        compatible(config, previous, provenance)
        model.load_state_dict(previous['model'])
        optimizer.load_state_dict(previous['optimizer'])
        restore_rng(previous['rng'])
        start, best, best_epoch, history = previous['epoch']+1, previous['best'], previous['best_epoch'], previous['history']
    elif (directory/'best.pt').exists():
        raise FileExistsError('best.pt without resumable last.pt')
    write_json(directory/'config.json', config.to_dict())
    write_json(directory/'data.json', provenance)
    if start > config.epochs:
        return directory/'best.pt'
    logger = Logger(directory, config.resolved_run_name, config.to_dict(), config.wandb_mode)
    try:
        for n in range(start, min(config.epochs, stop_after or config.epochs)+1):
            begun = time.monotonic()
            training = epoch(model, config, epoch_loader(datasets[0], config, n, True), target_device,
                             optimizer, config.max_train_batches)
            validation = epoch(model, config, epoch_loader(datasets[1], config), target_device,
                               maximum=config.max_val_batches)
            row = dict(epoch=n, seconds=time.monotonic()-begun,
                       **{f'train/{k}':v for k,v in training.items()},
                       **{f'val/{k}':v for k,v in validation.items()})
            history.append(row)
            improved = validation['total'] < best
            if improved:
                best, best_epoch = validation['total'], n
            checkpoint = dict(schema='lie_ssl_k1_v1', config=config.to_dict(), data=provenance,
                              model=model.state_dict(), optimizer=optimizer.state_dict(), rng=rng_state(),
                              epoch=n, best=best, best_epoch=best_epoch, history=history)
            save(checkpoint, last)
            if improved:
                save(checkpoint, directory/'best.pt')
            write_history_csv(directory/'history.csv', history)
            logger.log(row)
            print(f'{config.resolved_run_name} epoch={n} val={validation["total"]:.6f} seconds={row["seconds"]:.1f}', flush=True)
    finally:
        logger.close()
    return directory/'best.pt'
