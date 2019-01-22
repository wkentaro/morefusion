from __future__ import print_function

import inspect
import sys
import typing

import contextlib
import time


def find_caller(frame):
    co = frame.f_code
    func_name = co.co_name

    # Now extend the function name with class name, if available.
    try:
        class_name = frame.f_locals['self'].__class__.__name__
        func_name = '%s.%s' % (class_name, func_name)
    except KeyError:  # if the function is unbound, there is no self.
        pass

    return func_name


@contextlib.contextmanager
def timer(name: typing.Optional[str] = None) -> typing.Iterator:
    t0 = time.time()
    yield
    caller = find_caller(inspect.currentframe().f_back.f_back)
    if name is None:
        print(
            f'[INFO] [{caller}] elapsed time: {time.time() - t0} [s]',
            file=sys.stderr,
        )
    else:
        print(
            f'[INFO] [{caller}] [{name}] elapsed time: {time.time() - t0} [s]',
            file=sys.stderr,
        )
