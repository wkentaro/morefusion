import unittest

from objslampp.datasets.ycb_video import class_names
from objslampp.datasets.ycb_video import YCBVideoModels


class TestYCBVideoModelsDataset(unittest.TestCase):

    def setUp(self):
        self.dataset = YCBVideoModels()

    def test_init(self):
        assert hasattr(self.dataset, 'download')
        assert hasattr(self.dataset, 'get_model')
        assert hasattr(self.dataset, 'get_bbox_diagonal')

    def test_get_model(self):
        class_id = 2
        class_name = class_names[class_id]
        model = self.dataset.get_model(class_id=class_id)
        assert isinstance(model, dict)
        assert set(model.keys()) == {'textured_simple', 'points_xyz'}
        assert model == self.dataset.get_model(class_name=class_name)

    def test_get_bbox_diagonal(self):
        model = self.dataset.get_model(class_id=2)
        mesh_file = model['textured_simple']
        bbox_diagonal = self.dataset.get_bbox_diagonal(mesh_file=mesh_file)
        assert isinstance(bbox_diagonal, float)
