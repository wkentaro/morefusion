import chainer
from chainer.training import trigger as trigger_module


class ParameterTensorboardReport(chainer.training.Extension):

    def __init__(self, writer, trigger=(1, 'epoch')):
        self._writer = writer
        self._trigger = trigger_module.get_trigger(trigger)

    def __call__(self, trainer):
        if self._trigger(trainer):
            updater = trainer.updater
            optimizer = updater.get_optimizer('main')
            model = optimizer.target
            for name, param in model.namedparams():
                self._writer.add_histogram(
                    name, chainer.cuda.to_cpu(param.array), updater.iteration
                )
