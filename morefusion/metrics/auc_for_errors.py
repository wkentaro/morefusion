import numpy as np
import sklearn.metrics


def auc_for_errors(errors, max_threshold, *, nstep=1000, return_xy=False):
    errors = np.asarray(errors)

    assert errors.ndim == 1
    assert errors.min() >= 0, f"min of errors must be >=0: {errors.min()}"

    x = np.zeros((nstep,), dtype=float)
    y = np.zeros((nstep,), dtype=float)
    for i, threshold in enumerate(np.linspace(0, max_threshold, nstep)):
        accuracy = 1.0 * (errors <= threshold).sum() / errors.size
        x[i] = threshold
        y[i] = accuracy

    auc = sklearn.metrics.auc(x=x, y=y)
    auc_maximum = 1.0 * max_threshold
    auc = auc / auc_maximum  # scale to [0, 1]

    if return_xy:
        return auc, x, y
    else:
        return auc
