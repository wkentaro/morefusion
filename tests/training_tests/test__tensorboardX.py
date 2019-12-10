import shutil
import tempfile
import unittest

import tensorboardX

from morefusion.training._tensorboardX import SummaryWriterWithUpdater


class TestSummaryWriterWithUpdater(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        writer = tensorboardX.SummaryWriter(log_dir=self.tmp_dir)
        self.writer = SummaryWriterWithUpdater(writer=writer)

    def test_init(self):
        assert hasattr(self.writer, 'setup')
        assert hasattr(self.writer, 'scope')
        assert hasattr(self.writer, 'scoped')
        assert hasattr(self.writer, 'add_image')
        assert hasattr(self.writer, 'add_histogram')

    def test_setup(self):
        self.assertRaises(AttributeError, lambda: self.writer.global_step)

        class DummyUpdater(object):
            @property
            def iteration(self):
                return 0

        self.writer.setup(DummyUpdater())
        assert self.writer.global_step == 0

    def __del__(self):
        try:
            shutil.rmtree(self.tmp_dir)
        except IOError:
            pass
