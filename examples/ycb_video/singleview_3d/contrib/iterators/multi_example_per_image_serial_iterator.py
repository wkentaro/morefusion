import chainer
from chainer.iterators import _statemachine
import numpy


class MultiExamplePerImageSerialIterator(chainer.iterators.SerialIterator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._random_state = numpy.random.RandomState()

    def __next__(self):
        self._previous_epoch_detail = self.epoch_detail

        batch = []
        while True:
            self._state, indices = _statemachine.iterator_statemachine(
                state=self._state,
                batch_size=1,
                repeat=self.repeat,
                order_sampler=self.order_sampler,
                dataset_len=len(self.dataset),
            )
            if indices is None:
                raise StopIteration

            index = indices[0]
            examples = self.dataset[index]
            batch.extend(examples)
            if len(batch) >= self.batch_size:
                break

        indices = self._random_state.permutation(len(batch))[:self.batch_size]
        batch = [batch[index] for index in indices]

        return batch

    next = __next__
