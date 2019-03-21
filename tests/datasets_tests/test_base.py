import pathlib
import unittest

from objslampp.datasets.base import DatasetBase


class TestDatasetBase(unittest.TestCase):

    def test_no_override(self):
        dataset = DatasetBase()
        self.assertRaises(ValueError, lambda: len(dataset))
        self.assertRaises(ValueError, lambda: dataset.root_dir)
        self.assertRaises(ValueError, lambda: dataset.split)
        self.assertRaises(ValueError, lambda: dataset.ids)

    def test_override(self):

        class MyDataset(DatasetBase):
            def __init__(self):
                self._root_dir = pathlib.Path('/data/MyDataset')
                self._split = 'train'
                self._ids = ('video0/frame0', 'video0/frame1', 'video1/frame0')

        dataset = MyDataset()
        self.assertIsInstance(dataset.root_dir, pathlib.Path)
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.split, 'train')
        self.assertEqual(len(dataset.ids), 3)
        self.assertIsInstance(dataset.ids, tuple)
