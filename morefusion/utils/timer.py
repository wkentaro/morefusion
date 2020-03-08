import inspect
import types
import typing

import contextlib
import time


def find_caller(frame: types.FrameType):
    co = frame.f_code
    func_name = co.co_name

    # Now extend the function name with class name, if available.
    try:
        class_name = frame.f_locals["self"].__class__.__name__
        func_name = "%s.%s" % (class_name, func_name)
    except KeyError:  # if the function is unbound, there is no self.
        pass

    return func_name


@contextlib.contextmanager
def timer(name: typing.Optional[str] = None) -> typing.Iterator:
    t0 = time.time()
    yield

    frame = inspect.currentframe()
    if frame is None:
        caller = None
    else:
        caller = find_caller(frame.f_back.f_back)

    msg = "[INFO]"
    if caller:
        msg += f" [{caller}]"
    if name:
        msg += f" [{name}]"
    msg += f" elapsed time: {time.time() - t0} [s]"
    print(msg)
