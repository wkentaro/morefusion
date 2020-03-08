import chainer


class ParameterTensorboardReport(chainer.training.Extension):
    def __init__(self, writer):
        self._writer = writer

    def __call__(self, trainer):
        updater = trainer.updater
        optimizer = updater.get_optimizer("main")
        model = optimizer.target
        for name, param in model.namedparams():
            try:
                self._writer.add_histogram(
                    "/parameter" + name,
                    chainer.cuda.to_cpu(param.array),
                    updater.iteration,
                )
                self._writer.add_histogram(
                    "/gradient" + name,
                    chainer.cuda.to_cpu(param.grad),
                    updater.iteration,
                )
            except Exception:
                continue
