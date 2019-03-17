import contextlib


class SummaryWriterWithUpdater(object):

    def __init__(self, writer):
        self._writer = writer
        self._updater = None
        self._scope = None

    def setup(self, updater):
        self._updater = updater

    @contextlib.contextmanager
    def scope(self, value):
        self._scope = value
        yield
        self._scope = None

    def scoped(self, tag):
        if self._scope:
            return self._scope + '/' + tag
        else:
            return tag

    @property
    def global_step(self):
        if self._updater is None:
            raise AttributeError(
                'SummaryWriterWithUpdater.setup is not yet called'
            )
        return self._updater.iteration

    def add_image(self, tag, img_tensor, **kwargs):
        return self._writer.add_image(
            tag=self.scoped(tag),
            img_tensor=img_tensor,
            global_step=self.global_step,
            **kwargs
        )
