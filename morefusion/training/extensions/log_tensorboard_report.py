import json
import os
import shutil
import warnings

from chainer import reporter
from chainer import serializer as serializer_module
from chainer.training import extension
from chainer.training import trigger as trigger_module
from chainer import utils
from chainer import Variable
import six


class LogTensorboardReport(extension.Extension):
    def __init__(
        self,
        writer,
        keys=None,
        trigger=(1, "epoch"),
        postprocess=None,
        log_name="log",
    ):
        self._writer = writer
        self._keys = keys
        self._trigger = trigger_module.get_trigger(trigger)
        self._postprocess = postprocess
        self._log_name = log_name
        self._log = []

        self._init_summary()

    def __call__(self, trainer):
        # accumulate the observations
        keys = self._keys
        observation = trainer.observation
        summary = self._summary

        if keys is None:
            summary.add(observation)
        else:
            summary.add({k: observation[k] for k in keys if k in observation})

        for k, v in observation.items():
            if isinstance(v, Variable):
                v = v.array
            self._writer.add_scalar(k, float(v), trainer.updater.iteration)

        if trainer.updater.iteration == 0 or self._trigger(trainer):
            # output the result
            stats = self._summary.compute_mean()
            stats_cpu = {}
            for name, value in six.iteritems(stats):
                stats_cpu[name] = float(value)  # copy to CPU

            updater = trainer.updater
            stats_cpu["epoch"] = updater.epoch
            stats_cpu["iteration"] = updater.iteration
            stats_cpu["elapsed_time"] = trainer.elapsed_time

            if self._postprocess is not None:
                self._postprocess(stats_cpu)

            self._log.append(stats_cpu)

            # write to the log file
            if self._log_name is not None:
                log_name = self._log_name.format(**stats_cpu)
                with utils.tempdir(prefix=log_name, dir=trainer.out) as tempd:
                    path = os.path.join(tempd, "log.json")
                    with open(path, "w") as f:
                        json.dump(self._log, f, indent=4)

                    new_path = os.path.join(trainer.out, log_name)
                    shutil.move(path, new_path)

            # reset the summary for the next output
            self._init_summary()

    @property
    def log(self):
        """The current list of observation dictionaries."""
        return self._log

    def serialize(self, serializer):
        if hasattr(self._trigger, "serialize"):
            self._trigger.serialize(serializer["_trigger"])

        try:
            self._summary.serialize(serializer["_summary"])
        except KeyError:
            warnings.warn("The statistics are not saved.")

        # Note that this serialization may lose some information of small
        # numerical differences.
        if isinstance(serializer, serializer_module.Serializer):
            log = json.dumps(self._log)
            serializer("_log", log)
        else:
            log = serializer("_log", "")
            self._log = json.loads(log)

    def _init_summary(self):
        self._summary = reporter.DictSummary()
