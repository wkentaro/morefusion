import unittest

import numpy as np

from objslampp.datasets.ycb_video import class_names
from objslampp.datasets.ycb_video import YCBVideoModels


class TestYCBVideoModelsDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = YCBVideoModels()

    def test_init(self):
        assert hasattr(self.dataset, 'download')
        assert hasattr(self.dataset, 'get_model')
        assert hasattr(self.dataset, 'get_spherical_views')
        assert hasattr(self.dataset, 'get_bbox_diagonal')

    def test_get_model(self):
        class_id = 2
        class_name = class_names[class_id]
        model = self.dataset.get_model(class_id=class_id)
        assert isinstance(model, dict)
        assert set(model.keys()) == {'textured_simple', 'points_xyz'}
        assert model == self.dataset.get_model(class_name=class_name)

    def test_get_spherical_views(self):
        model = self.dataset.get_model(class_id=2)
        visual_file = model['textured_simple']
        angle_sampling = 5
        K, Ts_cam2world, rgbs, depths, segms = \
            self.dataset.get_spherical_views(
                visual_file=visual_file, angle_sampling=angle_sampling
            )
        n_viewpoints, H, W, _ = rgbs.shape

        assert n_viewpoints == 17
        assert (H, W) == (320, 320)

        assert isinstance(K, np.ndarray)
        assert K.shape == (3, 3)
        assert isinstance(Ts_cam2world, np.ndarray)
        assert Ts_cam2world.shape == (n_viewpoints, 4, 4)

        assert isinstance(rgbs, np.ndarray)
        assert rgbs.shape == (n_viewpoints, H, W, 3)
        assert rgbs.dtype == np.uint8

        assert isinstance(depths, np.ndarray)
        assert depths.shape == (n_viewpoints, H, W)
        assert isinstance(depths, np.ndarray)
        assert depths.dtype == np.float32

        assert isinstance(segms, np.ndarray)
        assert segms.shape == (n_viewpoints, H, W)
        assert segms.dtype == np.int32
        np.testing.assert_allclose(np.unique(segms), [-1, 0], atol=0, rtol=0)

        np.testing.assert_allclose(
            np.isnan(depths), segms == -1, atol=0, rtol=0
        )

    def test_get_bbox_diagonal(self):
        model = self.dataset.get_model(class_id=2)
        mesh_file = model['textured_simple']
        bbox_diagonal = self.dataset.get_bbox_diagonal(mesh_file=mesh_file)
        assert isinstance(bbox_diagonal, float)
