"""Dataset-wide component R²; raw regression outputs are never renormalized."""
import numpy as np


def rotation_metrics(target, prediction):
    target, prediction = np.asarray(target, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 2 or target.shape[1] != 2 or not len(target):
        raise ValueError('Expected nonempty matching [N, 2] arrays')
    if not (np.isfinite(target).all() and np.isfinite(prediction).all()):
        raise ValueError('Nonfinite rotation predictions')
    sse = np.sum((target-prediction)**2, axis=0)
    sst = np.sum((target-target.mean(axis=0))**2, axis=0)
    r2 = [float(1-sse[j]/sst[j]) if np.ptp(target[:,j]) > 0 and sst[j] > 0 else None for j in range(2)]
    true_angle = np.arctan2(target[:, 1], target[:, 0])
    predicted_angle = np.arctan2(prediction[:, 1], prediction[:, 0])
    error = np.abs((np.rad2deg(predicted_angle-true_angle)+180)%360-180)
    return dict(rotation_r2_cos=r2[0], rotation_r2_sin=r2[1],
                rotation_r2=None if None in r2 else float(np.mean(r2)),
                mae=float(error.mean()), rmse=float(np.sqrt(np.mean(error**2))),
                median=float(np.median(error)),
                **{f'within_{t}':float(np.mean(error <= t)) for t in (5,10,30)})
