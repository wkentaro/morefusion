import chainer.functions as F

import numpy as np


def _resize_image_float(x, output_shape):
    return F.resize_images(x[None], output_shape)[0].array


def resize_image(x, output_shape, order):
    assert x.ndim in {2, 3}

    # check order
    if order == 'HWC':
        x = x.transpose(2, 0, 1)  # HWC -> CHW
        y = resize_image(x, output_shape, order='CHW')
        y = y.transpose(1, 2, 0)  # CHW -> HWC
        return y
    elif order == 'HW':
        x = x[None, :, :]  # HW -> CHW
        y = resize_image(x, output_shape, order='CHW')
        y = y[0, :, :]     # CHW -> HW
        return y
    elif order == 'CHW':
        pass
    else:
        raise ValueError('unsupported order: {}'.format(order))

    # check dtype
    if np.issubdtype(x.dtype, np.floating):
        y = _resize_image_float(x, output_shape)
    elif x.dtype == np.uint8:
        x = x.astype(np.float32)
        y = _resize_image_float(x, output_shape)
        y = y.round().clip(0, 255).astype(np.uint8)
    elif x.dtype == bool:
        x = x.astype(np.float32)
        y = _resize_image_float(x, output_shape)
        y = y > 0.5
    else:
        raise TypeError('unsupported dtype: {}'.format(x.dtype))

    return y
