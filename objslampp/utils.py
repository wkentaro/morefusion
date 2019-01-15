import contextlib
import time


@contextlib.contextmanager
def timer(name):
    t0 = time.time()
    yield
    print(f'[{name}] elapsed time: {time.time() - t0} [s]')
