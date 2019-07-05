import unittest

from objslampp.datasets.ycb_video import YCBVideoModels


class TestYCBVideoModelsDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = YCBVideoModels()

    def test_init(self):
        assert hasattr(self.dataset, 'download')
        assert hasattr(self.dataset, 'get_bbox_diagonal')

    def test_get_bbox_diagonal(self):
        bbox_diagonal = self.dataset.get_bbox_diagonal(class_id=2)
        assert isinstance(bbox_diagonal, float)

    def test_get_voxel_pitch(self):
        pitch = self.dataset.get_voxel_pitch(dimension=32, class_id=2)
        assert isinstance(pitch, float)
