#!/usr/bin/env python

import argparse

from chainer.backends import cuda
import imgviz
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc

from objslampp.extra._cupy import resize_image


def main(gpu=-1):
    print('gpu:', gpu)

    # uint8
    img = scipy.misc.face()
    H, W = img.shape[:2]
    img_org = img.copy()
    if gpu >= 0:
        img = cuda.to_gpu(img)
    img = resize_image(img, (2 * H, 2 * W), order='HWC')
    img = cuda.to_cpu(img)

    assert img.shape == (2 * H, 2 * W, 3)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_org)
    plt.subplot(122)
    plt.imshow(img)
    plt.suptitle('uint8')
    plt.tight_layout()

    # float32
    img = scipy.misc.face()
    img_org = img.copy()
    img = img.astype(np.float32) / 255
    if gpu >= 0:
        img = cuda.to_gpu(img)
    img = resize_image(img, (2 * H, 2 * W), order='HWC')
    img = cuda.to_cpu(img)
    img = (img * 255).round().astype(np.uint8)

    assert img.shape == (2 * H, 2 * W, 3)
    plt.figure()
    plt.subplot(121)
    plt.imshow(img_org)
    plt.subplot(122)
    plt.imshow(img)
    plt.suptitle('float32')
    plt.tight_layout()

    # bool
    img = scipy.misc.face()
    gray = imgviz.rgb2gray(img)
    mask = gray > 127
    mask_org = mask.copy()
    if gpu >= 0:
        mask = cuda.to_gpu(mask)
    mask = cuda.to_cpu(mask)
    mask = resize_image(mask, (2 * H, 2 * W), order='HW')

    plt.figure()
    plt.subplot(121)
    plt.imshow(mask_org, cmap='gray')
    plt.subplot(122)
    plt.imshow(mask, cmap='gray')
    plt.suptitle('bool')
    plt.tight_layout()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gpu', type=int, default=-1, help='gpu id')
    args = parser.parse_args()

    main(gpu=args.gpu)
