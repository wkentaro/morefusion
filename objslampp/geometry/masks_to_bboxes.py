import numpy as np


def masks_to_bboxes(masks):
    """Convert mask images to bounding boxes.

    Parameters
    ----------
    masks: array
        Mask images.

    Returns
    -------
    bboxes: array of (y1, x1, y2, x2)
        Bounding boxes.
    """
    bboxes = np.zeros((len(masks), 4), dtype=np.float64)
    for i, mask in enumerate(masks):
        where = np.argwhere(mask)
        try:
            (y1, x1), (y2, x2) = where.min(0), where.max(0) + 1
        except ValueError:
            continue
        bboxes[i] = y1, x1, y2, x2
    return bboxes
