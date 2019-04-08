import numpy as np

from objslampp.geometry import masks_to_bboxes


def test_masks_to_bboxes():
    masks = np.random.randint(0, 2, (5, 16, 16), dtype=bool)
    bboxes = masks_to_bboxes(masks)

    assert isinstance(bboxes, np.ndarray)
    assert bboxes.dtype == np.float64

    mask = np.zeros((16, 16), dtype=bool)
    mask[6:9, 6:9] = True
    bbox = masks_to_bboxes([mask])[0]
    np.testing.assert_allclose(bbox, (6, 6, 9, 9))

    bbox = masks_to_bboxes(mask)
    np.testing.assert_allclose(bbox, (6, 6, 9, 9))
