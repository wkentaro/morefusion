import json
import os.path as osp

from chainer import training


class ArgsReport(training.Extension):

    def __init__(self, args, file_name='args'):
        if not isinstance(args, dict):
            args = args.__dict__

        self._args = args
        self._file_name = file_name
        self._flag_called = False

    def trigger(self, trainer):
        if self._flag_called:
            return False
        self._flag_called = True
        return True

    def __call__(self, trainer):
        with open(osp.join(trainer.out, self._file_name), 'w') as f:
            json.dump(self._args, f, indent=4)
