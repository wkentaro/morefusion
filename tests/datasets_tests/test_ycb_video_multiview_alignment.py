import unittest

import numpy as np
import pytest

from objslampp.datasets.ycb_video_multiview_alignment \
    import YCBVideoMultiViewAlignmentDataset


@pytest.mark.heavy
class TestYCBVideoMultiViewAlignment(unittest.TestCase):

    def setUp(self):
        self.dataset = YCBVideoMultiViewAlignmentDataset(
            split='val', class_ids=[2]
        )

    def test_init(self):
        assert hasattr(self.dataset, 'get_example')
        assert hasattr(self.dataset, 'get_ids')

        for image_id, class_id in self.dataset.ids[:10]:
            assert isinstance(image_id, str)
            assert isinstance(class_id, np.uint8)
            assert class_id == 2

    def test_get_example(self):
        example = self.dataset[0]
        assert set(example.keys()) == {
            'valid',
            'video_id',
            'class_id',
            'pitch',
            'cad_origin',
            'cad_rgbs',
            'cad_pcds',
            'cad_points',
            'scan_origin',
            'scan_rgbs',
            'scan_pcds',
            'scan_masks',
            'gt_pose',
            'gt_quaternion',
            'gt_translation',
        }

        assert isinstance(example['valid'], int)
        assert isinstance(example['video_id'], int)
        assert isinstance(example['class_id'], np.uint8)
        assert isinstance(example['pitch'], np.floating)

        assert isinstance(example['cad_origin'], np.ndarray)
        assert example['cad_origin'].shape == (3,)

        N, H, W, _ = example['cad_rgbs'].shape
        assert isinstance(example['cad_rgbs'], np.ndarray)
        assert example['cad_rgbs'].shape == (N, H, W, 3)
        assert example['cad_rgbs'].dtype == np.uint8
        assert isinstance(example['cad_pcds'], np.ndarray)
        assert example['cad_pcds'].shape == (N, H, W, 3)
        assert example['cad_pcds'].dtype == np.float32

        assert isinstance(example['scan_origin'], np.ndarray)
        assert example['scan_origin'].shape == (3,)

        N, H, W, _ = example['scan_rgbs'].shape
        assert isinstance(example['scan_rgbs'], np.ndarray)
        assert example['scan_rgbs'].shape == (N, H, W, 3)
        assert example['scan_rgbs'].dtype == np.uint8
        assert isinstance(example['scan_pcds'], np.ndarray)
        assert example['scan_pcds'].shape == (N, H, W, 3)
        assert example['scan_pcds'].dtype == np.float32

        assert isinstance(example['gt_pose'], np.ndarray)
        assert example['gt_pose'].dtype == np.float32
        assert example['gt_pose'].shape, (4, 4)
        assert isinstance(example['gt_quaternion'], np.ndarray)
        assert example['gt_quaternion'].dtype == np.float32
        assert example['gt_quaternion'].shape, (4,)
        assert isinstance(example['gt_translation'], np.ndarray)
        assert example['gt_translation'].dtype == np.float32
        assert example['gt_translation'].shape, (3,)
