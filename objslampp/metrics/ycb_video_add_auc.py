import numpy as np


# https://github.com/yuxng/YCB_Video_toolbox/blob/d08b645d406b93a988087fea42a5f6ac7330933c/plot_accuracy_keyframe.m#L65-L77  # NOQA
def ycb_video_add_auc(
    adds,
    *,
    max_value=0.1,
    return_xy=False
):
    adds = np.asarray(adds)

    assert adds.ndim == 1
    assert adds.min() >= 0, f'min of adds must be >=0: {adds.min()}'

    D = adds.copy()
    D[D > max_value] = np.inf
    d = np.sort(D)
    n = len(d)
    accuracy = np.cumsum(np.ones((1, n))) / n

    keep = np.isfinite(d)
    d = d[keep]
    accuracy = accuracy[keep]

    auc = VOCap(d, accuracy, max_value=max_value)

    if return_xy:
        x = np.r_[0, d, max_value]
        y = np.r_[0, accuracy, accuracy[-1]]
        return auc, x, y
    else:
        return auc


# https://github.com/yuxng/YCB_Video_toolbox/blob/d08b645d406b93a988087fea42a5f6ac7330933c/plot_accuracy_keyframe.m#L143-L155  # NOQA
def VOCap(rec, prec, max_value=0.1):
    # first append sentinel values at the end
    mrec = np.r_[0, rec, max_value]
    mpre = np.r_[0, prec, prec[-1]]

    for i in range(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i - 1]);

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.argwhere(mrec[1:] != mrec[:-1]) + 1

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i] - mrec[i - 1]) * mpre[i]) / max_value

    return ap
