"""Audit expected settings and prediction exports before producing comparisons."""
import csv
import json
from pathlib import Path
import numpy as np
from .config import setting
from .probe import metrics_from_rows
from .runtime import sha256,load,write_json,write_history_csv


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def compare(actual,expected):
    for key,value in actual.items():
        if value is None or expected[key] is None:
            if value is not expected[key]:
                raise AssertionError(f'Undefined metric mismatch: {key}')
        elif not np.isclose(value,expected[key],rtol=1e-6,atol=1e-7):
            raise AssertionError(f'Metric mismatch: {key}: {value} != {expected[key]}')


def coverage(rows,task,smoke):
    keys = [(int(r['source_index']),r.get('direction','single')) for r in rows]
    if len(set(keys)) != len(keys):
        raise AssertionError('Duplicate prediction identities')
    if not smoke:
        directions = ('i_to_j','j_to_i') if task=='rotation' else ('single',)
        if set(keys) != {(i,d) for i in range(10000) for d in directions}:
            raise AssertionError('Incomplete test-source/direction coverage')


def aggregate(root,data_root,smoke=False):
    root = Path(root)
    fits,rendering = [],[]
    per_digit_fits = []
    expected_seeds = [42] if smoke else list(range(42,47))
    for index in range(8):
        config = setting(index,data_root,root,smoke)
        directory = config.run_dir
        best,last = load(directory/'best.pt'),load(directory/'last.pt')
        if last['epoch'] != config.epochs or len(last['history']) != config.epochs:
            raise AssertionError('Incomplete backbone training')
        if best['epoch'] != last['best_epoch']:
            raise AssertionError('Incorrect backbone checkpoint selection')
        if not np.isclose(best['best'],min(r['val/total'] for r in last['history'])):
            raise AssertionError('Backbone did not select the validation minimum')
        digest = sha256(directory/'best.pt')
        report = json.loads((directory/'evaluation/result.json').read_text())
        if report['checkpoint_sha256'] != digest:
            raise AssertionError('Rendering checkpoint mismatch')
        rows = read_csv(directory/'evaluation/per_source.csv')
        coverage(rows,'rotation',smoke)
        compare({k:float(np.mean([float(r[k]) for r in rows])) for k in report['metrics']},report['metrics'])
        rendering.append(dict(run_name=config.resolved_run_name,**report['metrics'],
            periodicity_residual=report['diagnostics']['periodicity_residual']))
        for task in ('classification','rotation'):
            for representation in ('canonical','pose'):
                for seed in expected_seeds:
                    probe_dir = directory/'probes'/task/representation/f'seed_{seed}'
                    result = json.loads((probe_dir/'result.json').read_text())
                    probe_last = load(probe_dir/'last.pt')
                    if probe_last['epoch'] != (2 if smoke else 100):
                        raise AssertionError('Incomplete probe training')
                    score_key = 'val/mae' if task=='rotation' else 'val/top1'
                    expected_score = min((r[score_key] if task=='rotation' else -r[score_key]) for r in probe_last['history'])
                    if not np.isclose(expected_score,result['best_validation_score']):
                        raise AssertionError('Probe validation selection mismatch')
                    if result['checkpoint_sha256'] != digest or not result['backbone_frozen']:
                        raise AssertionError('Probe checkpoint/freeze mismatch')
                    rows = read_csv(probe_dir/'predictions.csv')
                    coverage(rows,task,smoke)
                    for row in rows:
                        for key in ('target_cos','target_sin','predicted_cos','predicted_sin'):
                            if key in row:
                                row[key] = float(row[key])
                    measured = metrics_from_rows(task,rows)
                    compare(measured,result['test'])
                    fit = dict(run_name=config.resolved_run_name,task=task,representation=representation,seed=seed,**measured)
                    fits.append(fit)
                    digit_rows = read_csv(probe_dir/'per_digit.csv')
                    for row in digit_rows:
                        group = [r for r in rows if int(r['digit'])==int(row['digit'])]
                        values = metrics_from_rows(task,group)
                        compare(values,{k:float(row[k]) if row[k] else None for k in values})
                        per_digit_fits.append(dict(run_name=config.resolved_run_name,task=task,
                            representation=representation,seed=seed,digit=int(row['digit']),**values))
    output = root/'comparison'
    output.mkdir(parents=True,exist_ok=True)
    write_history_csv(output/'rendering.csv',rendering)
    # Tasks have different metric columns, so keep separate flat CSVs.
    summaries = []
    for task in ('classification','rotation'):
        task_rows = [r for r in fits if r['task']==task]
        write_history_csv(output/f'{task}_fits.csv',task_rows)
        write_history_csv(output/f'{task}_per_digit.csv',[r for r in per_digit_fits if r['task']==task])
        metrics = [k for k in task_rows[0] if k not in ('run_name','task','representation','seed')]
        for name,rep in sorted({(r['run_name'],r['representation']) for r in task_rows}):
            group = [r for r in task_rows if r['run_name']==name and r['representation']==rep]
            summary = dict(run_name=name,task=task,representation=rep,probe_seeds=len(group),backbone_seed=42)
            for metric in metrics:
                values = [r[metric] for r in group]
                summary[metric+'_mean'] = None if None in values else float(np.mean(values))
                summary[metric+'_std'] = None if None in values else float(np.std(values,ddof=1 if len(values)>1 else 0))
            summaries.append(summary)
        write_history_csv(output/f'{task}_summary.csv',[r for r in summaries if r['task']==task])
    write_json(output/'summary.json',dict(rendering=rendering,probes=summaries,
        uncertainty='Standard deviations are across probe seeds, not backbone training seeds.'))
    write_json(output/'audit.json',dict(passed=True,backbones=8,probe_fits=len(fits),smoke=smoke,
        r2_definition='Uniform mean of raw sine/cosine component R² over all test pairs and both directions'))
    rotation = [r for r in summaries if r['task']=='rotation']
    lines = ['# MNIST K=1 comparison','',
        'R² is the uniform mean of raw cosine/sine component scores. Standard deviations measure probe-seed variability; each backbone uses seed 42.','',
        '| Setting | Representation | Rotation R² | Circular MAE (°) |',
        '|---|---|---:|---:|']
    for row in rotation:
        score = row['rotation_r2_mean']
        r2 = 'undefined' if score is None else f'{score:.4f} ± {row["rotation_r2_std"]:.4f}'
        lines.append(f'| {row["run_name"]} | {row["representation"]} | {r2} | {row["mae_mean"]:.2f} ± {row["mae_std"]:.2f} |')
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    print(f'Audit passed: 8 backbones, {len(fits)} probe fits',flush=True)
