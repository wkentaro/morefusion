import numpy as np

from chainer.dataset import DatasetMixin


class RandomSamplingDataset(DatasetMixin):

    def __init__(self, dataset, n_sample, seed=0):
        self._dataset = dataset
        self._n_sample = n_sample
        self._random_state = np.random.RandomState(seed)

    def __len__(self):
        return self._n_sample

    def get_example(self, index):
        del index

        index = self._random_state.randint(0, len(self._dataset))
        return self._dataset.get_example(index)
