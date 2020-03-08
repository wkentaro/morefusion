import collections
import copy
import os.path as osp
import re
import warnings

import chainer
from chainer.dataset import convert as convert_module
from chainer import function
import chainer.reporter as reporter_module
from chainer.training.extensions.evaluator import _IteratorProgressBar
import numpy as np
import pandas

from ... import metrics


class PoseEstimationEvaluator(chainer.training.extensions.Evaluator):
    @property
    def comm(self):
        if not hasattr(self, "_comm"):
            self._comm = None
        return self._comm

    @comm.setter
    def comm(self, value):
        self._comm = value

    def evaluate(self):
        iterator = self._iterators["main"]
        eval_func = self.eval_func or self._targets["main"]

        if self.eval_hook:
            self.eval_hook(self)

        if hasattr(iterator, "reset"):
            iterator.reset()
            it = iterator
        else:
            warnings.warn(
                "This iterator does not have the reset method. Evaluator "
                "copies the iterator instead of resetting. This behavior is "
                "deprecated. Please implement the reset method.",
                DeprecationWarning,
            )
            it = copy.copy(iterator)

        if self._progress_bar and self.comm is None or self.comm.rank == 0:
            pbar = _IteratorProgressBar(iterator=it)

        observations = []
        for batch in it:
            observation = {}
            with reporter_module.report_scope(observation):
                in_arrays = convert_module._call_converter(
                    self.converter, batch, self.device
                )
                with function.no_backprop_mode():
                    if isinstance(in_arrays, tuple):
                        eval_func(*in_arrays)
                    elif isinstance(in_arrays, dict):
                        eval_func(**in_arrays)
                    else:
                        eval_func(in_arrays)

            for k, v in list(observation.items()):
                if hasattr(v, "array"):
                    v = chainer.cuda.to_cpu(v.array)
                if hasattr(v, "item"):
                    v = v.item()
                observation[k] = v
            observations.append(observation)

            if self._progress_bar and self.comm is None or self.comm.rank == 0:
                pbar.update()

        if self._progress_bar and self.comm is None or self.comm.rank == 0:
            pbar.close()

        local_df = pandas.DataFrame(observations)
        if self.comm:
            dfs = self.comm.gather_obj(local_df)
            if self.comm.rank == 0:
                global_df = pandas.concat(dfs, sort=True)
            else:
                return {}
        else:
            global_df = local_df

        summary = reporter_module.DictSummary()
        adds = collections.defaultdict(list)
        for _, row in global_df.iterrows():
            observation = row.dropna().to_dict()

            observation_processed = {}
            add_types = ["add", "add_s", "add_or_add_s"]
            for key, value in observation.items():
                for add_type in add_types:
                    # validation/main/{add_type}/{class_id}/{instance_id}
                    pattern = f"validation/main/{add_type}/([0-9]+)/.+"
                    match = re.match(pattern, key)
                    if not match:
                        continue
                    class_id = match.groups()[0]
                    key = f"validation/main/{add_type}/{class_id}"
                    adds[f"{add_type}/{class_id}"].append(value)
                    break
                observation_processed[key] = value
            summary.add(observation_processed)
        result = summary.compute_mean()

        # compute auc for adds
        for add_type_and_class_id, values in adds.items():
            # auc = metrics.auc_for_errors(values, max_threshold=0.1)
            auc = metrics.ycb_video_add_auc(values, max_value=0.1)
            result[f"validation/main/auc/{add_type_and_class_id}"] = auc
            lt_2cm = (np.array(values) < 0.02).sum() / len(values)
            result[f"validation/main/<2cm/{add_type_and_class_id}"] = lt_2cm

        # average child observations
        parent_keys = [
            "validation/main/loss",
            "validation/main/loss_quaternion",
            "validation/main/loss_translation",
            "validation/main/add",
            "validation/main/add_s",
            "validation/main/add_or_add_s",
            "validation/main/auc/add",
            "validation/main/auc/add_s",
            "validation/main/auc/add_or_add_s",
            "validation/main/<2cm/add",
            "validation/main/<2cm/add_s",
            "validation/main/<2cm/add_or_add_s",
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
