import numpy as np


def masks_to_bboxes(masks):
    """Convert mask images to bounding boxes.

    Parameters
    ----------
    masks: array (N, H, W) or (H, W)
        Mask images.

    Returns
    -------
    bboxes: array of (y1, x1, y2, x2) or (y1, x1, y2, x2)
        Bounding boxes.
    """
    masks = np.asarray(masks)
    assert masks.dtype == bool

    ndim = masks.ndim
    assert ndim in [2, 3], "masks must be 2 or 3 dimensional"

    if ndim == 2:
        masks = masks[None]

    bboxes = np.zeros((len(masks), 4), dtype=np.float64)
    for i, mask in enumerate(masks):
        where = np.argwhere(mask)
        try:
            (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        except ValueError:
            continue
        bboxes[i] = y1, x1, y2, x2

    if ndim == 2:
        return bboxes[0]
    else:
        return bboxes
