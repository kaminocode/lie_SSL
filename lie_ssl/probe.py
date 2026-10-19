"""Frozen-backbone digit and ordered-pair rotation probes."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from .rotation_metrics import rotation_metrics
from .runtime import (device, frozen_model, build_datasets, data_provenance, seed_everything,
    epoch_loader, limited_batches, state_copy, assert_frozen, sha256, save, load,
    rng_state, restore_rng, Logger, write_json, write_history_csv)


def make_head(task, width):
    if task == 'classification':
        return nn.Linear(width, 10)
    return nn.Sequential(nn.Linear(2*width,1024), nn.ReLU(), nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024,2))


def relative_target(first, second):
    delta = (second-first+180)%360-180
    radians = torch.deg2rad(delta)
    return torch.stack((radians.cos(), radians.sin()), -1)


def metrics_from_rows(task, rows):
    if task == 'rotation':
        return rotation_metrics([[r['target_cos'],r['target_sin']] for r in rows],
                                [[r['predicted_cos'],r['predicted_sin']] for r in rows])
    return dict(top1=float(np.mean([int(r['predicted_digit']) == int(r['digit']) for r in rows])),
                top5=float(np.mean([int(r['top5_correct']) for r in rows])))


def epoch(model, head, representation, task, loader, target_device, optimizer=None, maximum=0, collect=False):
    model.eval()
    head.train(optimizer is not None)
    total, samples, top1, top5 = 0., 0, 0, 0
    targets, outputs, rows = [], [], []
    for batch in limited_batches(loader, maximum):
        first = batch['x_i'].to(target_device, non_blocking=True)
        labels = batch['label'].to(target_device)
        with torch.no_grad():
            features = model.representation(first, representation)
            if task == 'rotation':
                other = model.representation(batch['x_j'].to(target_device, non_blocking=True), representation)
                features = torch.cat((torch.cat((features,other),1), torch.cat((other,features),1)))
                ai, aj = batch['angle_i'].to(target_device), batch['angle_j'].to(target_device)
                target = torch.cat((relative_target(ai,aj), relative_target(aj,ai)))
        with torch.set_grad_enabled(optimizer is not None):
            predicted = head(features)
            loss = F.cross_entropy(predicted,labels) if task == 'classification' else F.mse_loss(predicted,target)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite probe loss')
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in head.parameters()):
                raise FloatingPointError('Nonfinite probe gradient')
            optimizer.step()
        count = len(predicted)
        total += float(loss.detach())*count
        samples += count
        if task == 'rotation':
            truth, pred = target.cpu().numpy(), predicted.detach().cpu().numpy()
            targets.append(truth)
            outputs.append(pred)
        else:
            ranked = predicted.detach().topk(5,dim=1).indices
            top1 += int((ranked[:,0] == labels).sum())
            top5 += int(ranked.eq(labels[:,None]).any(dim=1).sum())
        if collect:
            for i in range(count):
                j = i % len(first)
                row = dict(source_index=int(batch['source_index'][j]), digit=int(batch['label'][j]))
                if task == 'rotation':
                    angle = np.rad2deg(np.arctan2(pred[i,1],pred[i,0]))
                    target_angle = np.rad2deg(np.arctan2(truth[i,1],truth[i,0]))
                    row.update(direction='i_to_j' if i < len(first) else 'j_to_i',
                               target_cos=float(truth[i,0]), target_sin=float(truth[i,1]),
                               predicted_cos=float(pred[i,0]), predicted_sin=float(pred[i,1]),
                               target_degrees=float(target_angle), predicted_degrees=float(angle))
                else:
                    row.update(predicted_digit=int(ranked[i,0]), top5_correct=int(ranked[i].eq(labels[i]).any()))
                rows.append(row)
    if not samples:
        raise ValueError('Empty probe split')
    metrics = dict(loss=total/samples,samples=samples)
    metrics.update(rotation_metrics(np.concatenate(targets),np.concatenate(outputs)) if task == 'rotation'
                   else dict(top1=top1/samples,top5=top5/samples))
    return metrics, rows


def fit(checkpoint, task, representation, seed, epochs=100, smoke=False, stop_after=None, target_device=None):
    target_device = target_device or device()
    model, config, payload = frozen_model(checkpoint, target_device)
    datasets = build_datasets(config, two_view=task=='rotation')
    if data_provenance(config,datasets) != payload['data']:
        raise ValueError('Probe data provenance mismatch')
    seed_everything(seed)
    width = config.latent_dim if representation=='canonical' else (2 if config.circle else 1)
    head = make_head(task,width).to(target_device)
    optimizer = torch.optim.Adam(head.parameters(),lr=.001)
    directory = Path(checkpoint).parent/'probes'/task/representation/f'seed_{seed}'
    directory.mkdir(parents=True,exist_ok=True)
    signature = dict(checkpoint_sha256=sha256(checkpoint), task=task, representation=representation,
                     seed=seed, epochs=epochs, width=width, lr=.001, smoke=smoke, protocol=config.protocol)
    last = directory/'last.pt'
    start, best, best_epoch, history = 1, float('inf'), 0, []
    if last.exists():
        previous = load(last)
        if previous['signature'] != signature:
            raise ValueError('Incompatible probe resume')
        head.load_state_dict(previous['head'])
        optimizer.load_state_dict(previous['optimizer'])
        restore_rng(previous['rng'])
        start,best,best_epoch,history = previous['epoch']+1,previous['best'],previous['best_epoch'],previous['history']
        if start > epochs and (directory/'result.json').exists():
            return directory
    before = state_copy(model)
    logger = Logger(directory, f'{config.resolved_run_name}-{task}-{representation}-{seed}',signature,config.wandb_mode)
    maximum = 2 if smoke else 0
    try:
        for n in range(start,min(epochs,stop_after or epochs)+1):
            training,_ = epoch(model,head,representation,task,
                epoch_loader(datasets[0],config,n,True,seed),target_device,optimizer,maximum)
            validation,_ = epoch(model,head,representation,task,
                epoch_loader(datasets[1],config,seed=seed),target_device,maximum=maximum)
            row = dict(epoch=n,**{f'train/{k}':v for k,v in training.items()},**{f'val/{k}':v for k,v in validation.items()})
            history.append(row)
            score = validation['mae'] if task=='rotation' else -validation['top1']
            improved = score < best
            if improved:
                best,best_epoch = score,n
            state = dict(signature=signature,head=head.state_dict(),optimizer=optimizer.state_dict(),
                         rng=rng_state(),epoch=n,best=best,best_epoch=best_epoch,history=history)
            save(state,last)
            if improved:
                save(state,directory/'best.pt')
            write_history_csv(directory/'history.csv',history)
            logger.log(row)
            print(f'{config.resolved_run_name} {task}/{representation}/s{seed} epoch={n} score={score:.5f}',flush=True)
        if stop_after and stop_after < epochs:
            return directory
        head.load_state_dict(load(directory/'best.pt')['head'])
        test,rows = epoch(model,head,representation,task,epoch_loader(datasets[2],config,seed=seed),
                         target_device,maximum=maximum,collect=True)
        assert_frozen(model,before)
        per_digit = []
        for digit in sorted({r['digit'] for r in rows}):
            group = [r for r in rows if r['digit']==digit]
            per_digit.append(dict(digit=digit,samples=len(group),**metrics_from_rows(task,group)))
        write_history_csv(directory/'predictions.csv',rows)
        write_history_csv(directory/'per_digit.csv',per_digit)
        result = dict(**signature,run_name=config.resolved_run_name,test=test,best_epoch=best_epoch,
                      best_validation_score=best,backbone_frozen=True,backbone_seed=config.seed,
                      selected_by='validation_circular_mae' if task=='rotation' else 'validation_top1')
        write_json(directory/'result.json',result)
        logger.log({f'test/{k}':v for k,v in test.items()})
        logger.log({f'test/digit_{row["digit"]}/{k}':v for row in per_digit for k,v in row.items() if k not in ('digit','samples')})
        plot_predictions(task,rows,directory)
    finally:
        logger.close()
    return directory


def plot_predictions(task,rows,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(5,4))
    if task=='rotation':
        ax.hist2d([r['target_degrees'] for r in rows],[r['predicted_degrees'] for r in rows],bins=72,
                  range=[[-180,180],[-180,180]])
        ax.set(xlabel='True relative angle (degrees)',ylabel='Predicted relative angle (degrees)')
    else:
        confusion = np.zeros((10,10),dtype=int)
        for r in rows:
            confusion[r['digit'],r['predicted_digit']] += 1
        ax.imshow(confusion)
        ax.set(xlabel='Predicted digit',ylabel='True digit',xticks=range(10),yticks=range(10))
        np.savetxt(directory/'confusion_matrix.csv',confusion,fmt='%d',delimiter=',')
    fig.tight_layout()
    fig.savefig(directory/'predictions.png',dpi=140)
    fig.savefig(directory/'predictions.pdf')
    plt.close(fig)
