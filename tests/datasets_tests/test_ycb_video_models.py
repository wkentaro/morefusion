import unittest

from objslampp.datasets.ycb_video import class_names
from objslampp.datasets.ycb_video import YCBVideoModels


class TestYCBVideoModelsDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = YCBVideoModels()

    def test_init(self):
        assert hasattr(self.dataset, 'download')
        assert hasattr(self.dataset, 'get_model_files')
        assert hasattr(self.dataset, 'get_bbox_diagonal')

    def test_get_model_files(self):
        class_id = 2
        class_name = class_names[class_id]
        files = self.dataset.get_model_files(class_id=class_id)
        assert isinstance(files, dict)
        assert set(files.keys()) == {
            'textured_simple',
            'points_xyz',
            'solid_binvox',
        }
        assert files == self.dataset.get_model_files(class_name=class_name)

    def test_get_bbox_diagonal(self):
        bbox_diagonal = self.dataset.get_bbox_diagonal(class_id=2)
        assert isinstance(bbox_diagonal, float)

    def test_get_voxel_pitch(self):
        pitch = self.dataset.get_voxel_pitch(dimension=32, class_id=2)
        assert isinstance(pitch, float)
