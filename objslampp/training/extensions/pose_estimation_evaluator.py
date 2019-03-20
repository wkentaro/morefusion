import collections
import copy
import os.path as osp
import warnings

import chainer
from chainer import function
from chainer.training.extensions import util
import chainer.reporter as reporter_module

from ... import metrics


class PoseEstimationEvaluator(chainer.training.extensions.Evaluator):

    def evaluate(self):
        iterator = self._iterators['main']
        eval_func = self.eval_func or self._targets['main']

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, 'reset'):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                'This iterator does not have the reset method. Evaluator '
                'copies the iterator instead of resetting. This behavior is '
                'deprecated. Please implement the reset method.',
                DeprecationWarning)
            it = copy.copy(iterator)

        summary = reporter_module.DictSummary()

        if self._progress_bar:
            pbar = util.IteratorProgressBar(iterator=it, title='validation ')

        adds = collections.defaultdict(list)
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = self._call_converter(batch, self.device)
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            observation_processed = {}
            parent_keys = [
                'validation/main/add',
                'validation/main/add_rotation',
            ]
            for key, value in observation.items():
                if osp.dirname(key) in parent_keys:
                    # add/2, add_rotation/2
                    sub_key = key[len('validation/main/'):]
                    adds[sub_key].append(value)
                else:
                    observation_processed[key] = value

            summary.add(observation_processed)

            if self._progress_bar:
                pbar.update()

        if self._progress_bar:
            pbar.close()

        result = summary.compute_mean()

        # compute auc for adds
        for sub_key, values in adds.items():
            auc = metrics.auc_for_errors(values, max_threshold=0.1)
            result[f'validation/main/auc/{sub_key}'] = auc

        # average child observations
        parent_keys = [
            'validation/main/loss',
            'validation/main/loss_quaternion',
            'validation/main/loss_translation',
            'validation/main/auc/add',
            'validation/main/auc/add_rotation',
        ]
        summary = reporter_module.DictSummary()
        for parent_key in parent_keys:
            if parent_key in result:
                continue
            for key, value in result.items():
                if osp.dirname(key) == parent_key:
                    summary.add({parent_key: value})
        result.update(summary.compute_mean())

        return result
