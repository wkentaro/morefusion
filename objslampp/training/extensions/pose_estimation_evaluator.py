import collections
import copy
import os.path as osp
import re
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
            add_types = ['add', 'add_s', 'addr', 'addr_s']
            for key, value in observation.items():
                for add_type in add_types:
                    # validation/main/{add_type}/{class_id}/{instance_id}
                    pattern = f'validation/main/{add_type}/([0-9]+)/[0-9]+'
                    match = re.match(pattern, key)
                    if not match:
                        continue
                    class_id = match.groups()[0]
                    key = f'validation/main/{add_type}/{class_id}'
                    adds[f'{add_type}/{class_id}'].append(value)
                    break
                observation_processed[key] = value
            summary.add(observation_processed)

            if self._progress_bar:
                pbar.update()

        if self._progress_bar:
            pbar.close()

        result = summary.compute_mean()

        # compute auc for adds
        for add_type_and_class_id, values in adds.items():
            # auc = metrics.auc_for_errors(values, max_threshold=0.1)
            auc = metrics.ycb_video_add_auc(values, max_value=0.1)
            result[f'validation/main/auc/{add_type_and_class_id}'] = auc

        # average child observations
        parent_keys = [
            'validation/main/loss',
            'validation/main/loss_quaternion',
            'validation/main/loss_translation',
            'validation/main/add',
            'validation/main/add_s',
            'validation/main/addr',
            'validation/main/addr_s',
            'validation/main/auc/add',
            'validation/main/auc/add_s',
            'validation/main/auc/addr',
            'validation/main/auc/addr_s',
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
