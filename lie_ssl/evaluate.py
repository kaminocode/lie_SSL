"""Matched rendering metrics, fixed-source sweeps, and action diagnostics."""
from pathlib import Path
import math
import numpy as np
import torch
from .image_metrics import image_metrics, METRIC_SETTINGS
from .model import decode_with_action
from .runtime import (device,frozen_model,build_datasets,data_provenance,epoch_loader,
    limited_batches,sha256,write_json,write_history_csv,Logger)


@torch.no_grad()
def evaluate(checkpoint,target_device=None):
    target_device = target_device or device()
    model,config,payload = frozen_model(checkpoint,target_device)
    datasets = build_datasets(config)
    if data_provenance(config,datasets) != payload['data']:
        raise ValueError('Evaluation data provenance mismatch')
    directory = Path(checkpoint).parent/'evaluation'
    directory.mkdir(parents=True,exist_ok=True)
    rows, coefficients, grids = [],[],None
    for batch in limited_batches(epoch_loader(datasets[2],config),config.max_test_batches):
        first,second = (batch[k].to(target_device) for k in ('x_i','x_j'))
        left,right = model(first),model(second)
        transfer_right = decode_with_action(model,left['z'],right['A'])
        transfer_left = decode_with_action(model,right['z'],left['A'])
        if grids is None:
            grids = torch.stack((first[:10],left['observed_recon'][:10],second[:10],transfer_right[:10]),1).cpu()
        coefficients.extend((left['pose_coefficient'].cpu().numpy(),right['pose_coefficient'].cpu().numpy()))
        for direction,source,target,reconstruction,transfer in (
            ('i_to_j',first,second,right['observed_recon'],transfer_right),
            ('j_to_i',second,first,left['observed_recon'],transfer_left)):
            rec,swap = image_metrics(reconstruction,target),image_metrics(transfer,target)
            for i in range(len(source)):
                rows.append(dict(source_index=int(batch['source_index'][i]),digit=int(batch['label'][i]),
                    direction=direction,**{f'reconstruction_{k}':float(v[i]) for k,v in rec.items()},
                    **{f'transfer_{k}':float(v[i]) for k,v in swap.items()}))
    if not rows:
        raise ValueError('Empty evaluation')
    scores = {k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k not in ('source_index','digit','direction')}
    # Small diagnostic matrices use CPU LAPACK; this ROCm build has no MAGMA eig.
    B = model.action.B.detach().cpu().double()
    eigenvalues = torch.linalg.eigvals(B)
    values = np.concatenate(coefficients).ravel()
    counts,edges = np.histogram(values,bins=36,range=(-math.pi,math.pi) if config.circle else None)
    diagnostics = dict(frobenius_norm=float(torch.linalg.matrix_norm(B,ord='fro')),
        spectral_norm=float(torch.linalg.matrix_norm(B,ord=2)),
        skew_residual=float(torch.linalg.matrix_norm(B+B.T)),
        periodicity_residual=float(torch.linalg.matrix_norm(torch.matrix_exp(2*math.pi*B)-torch.eye(len(B),device=B.device))),
        eigenvalues_real=eigenvalues.real.cpu().tolist(),eigenvalues_imag=eigenvalues.imag.cpu().tolist(),
        coefficient_quantiles=np.quantile(values,[0,.01,.25,.5,.75,.99,1]).tolist(),
        coefficient_histogram=dict(counts=counts.tolist(),edges=edges.tolist()),
        circle_dense_seam_warning=config.circle and not config.structured,
        block_spectral_norms=[float(torch.linalg.matrix_norm(B[j:j+2,j:j+2],ord=2)) for j in range(0,len(B),2)]
            if config.structured else None)
    write_history_csv(directory/'per_source.csv',rows)
    result = dict(run_name=config.resolved_run_name,checkpoint_sha256=sha256(checkpoint),
                  best_epoch=payload['epoch'],samples=len(rows),metrics=scores,diagnostics=diagnostics,
                  metric_settings=METRIC_SETTINGS,data=payload['data'])
    write_json(directory/'result.json',result)
    np.savez_compressed(directory/'rendering_pixels.npz',grid=grids.numpy(),coefficients=values,generator=B.cpu().numpy())
    plot_grid(grids.reshape(-1,1,28,28).numpy(),len(grids),4,directory/'reconstruction_transfer',
              'Source / self reconstruction / target / transfer')
    sweep_rows,provenance = [],[]
    dataset = datasets[2]
    angles = list(range(0,360,30))
    for digit in range(10):
        index = int(torch.nonzero(dataset.labels==digit)[0])
        target_images,blocks = dataset.orbit(index,angles)
        source = dataset.render(index,45.).unsqueeze(0).to(target_device)
        source_output = model(source)
        target_output = model(target_images.to(target_device))
        transfer = decode_with_action(model,source_output['z'].expand(12,-1),target_output['A']).cpu()
        sweep_rows.extend((target_images,transfer))
        provenance.append(dict(digit=digit,source_index=int(dataset.source_indices[index]),
                               source_angle=45,angles=angles,block_corners=blocks.tolist()))
    sweeps = torch.cat(sweep_rows).numpy()
    np.savez_compressed(directory/'sweep_pixels.npz',images=sweeps)
    write_json(directory/'sweep_provenance.json',provenance)
    plot_grid(sweeps,20,12,directory/'rotation_sweeps','True / transferred rotations for digits 0–9')
    logger = Logger(directory,config.resolved_run_name+'-evaluation',config.to_dict(),config.wandb_mode)
    try:
        logger.log({f'test/{k}':v for k,v in scores.items()})
        logger.log({f'generator/{k}':diagnostics[k] for k in ('frobenius_norm','spectral_norm','skew_residual','periodicity_residual')})
    finally:
        logger.close()
    return result


def plot_grid(images,nrows,ncols,path,title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(nrows,ncols,figsize=(ncols*.8,nrows*.8),squeeze=False)
    for ax,image in zip(axes.ravel(),images):
        ax.imshow(image.squeeze(),cmap='gray',vmin=0,vmax=1)
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout(pad=.15,rect=(0,0,1,.98))
    fig.savefig(path.with_suffix('.png'),dpi=130)
    fig.savefig(path.with_suffix('.pdf'))
    plt.close(fig)
