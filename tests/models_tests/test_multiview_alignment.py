import unittest

from objslampp.models import MultiViewAlignmentModel


class TestMultiViewAlignmentModel(unittest.TestCase):

    def setUp(self):
        self.model = MultiViewAlignmentModel(extractor='vgg16')

    def test_init(self):
        assert hasattr(self.model, '__call__')
        assert hasattr(self.model, 'encode')
        assert hasattr(self.model, 'predict_from_code')
        assert hasattr(self.model, 'predict')
        assert hasattr(self.model, 'loss')
