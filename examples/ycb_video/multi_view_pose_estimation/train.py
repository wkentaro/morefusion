#!/usr/bin/env python

import argparse
import datetime
import pathlib
import pprint
import random
import socket
import textwrap

import chainer
import numpy as np
import termcolor

import objslampp


here = pathlib.Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    msg = f"==> Started training"
    termcolor.cprint(msg, attrs={'bold': True})

    now = datetime.datetime.now()
    args.timestamp = now.isoformat()
    args.out = str(here / 'logs' / now.strftime('%Y%m%d_%H%M%S.%f'))
    args.hostname = socket.gethostname()

    msg = pprint.pformat(args.__dict__)
    msg = textwrap.indent(msg, prefix=' ' * 2)
    msg = f'==> Arguments:\n\n{msg}\n'
    termcolor.cprint(msg, attrs={'bold': True})

    # -------------------------------------------------------------------------

    # device initialization
    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()

    # seed initialization
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.gpu >= 0:
        chainer.cuda.cupy.random.seed(args.seed)

    # dataset initialization
    data_train = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
        'train'
    )
    # data_valid = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
    #     'val'
    # )

    # model initialization
    model = objslampp.models.SimpleMV3DCNNModel()
    if args.gpu >= 0:
        model.to_gpu()

    # optimizer initialization
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # link.disable_update()?

    # chainer.datasets.TransformDataset?

    # iterator initialization
    iter_train = chainer.iterators.SerialIterator(
        data_train, batch_size=1, repeat=True, shuffle=True
    )
    # iter_valid = chainer.iterators.SerialIterator(
    #     data_valid, batch_size=1, repeat=False, shuffle=False
    # )

    updater = chainer.training.updater.StandardUpdater(
        iterator=iter_train,
        optimizer=optimizer,
        # converter=my_converter,
        device=args.gpu,
    )

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(updater, (10, 'epoch'), out=args.out)

    log_interval = 20, 'iteration'

    # logging
    trainer.extend(
        chainer.training.extensions.observe_lr(),
        trigger=log_interval,
    )
    trainer.extend(chainer.training.extensions.LogReport(trigger=log_interval))
    trainer.extend(
        chainer.training.extensions.PrintReport(
            [
                'epoch',
                'iteration',
                'elapsed_time',
                'lr',
                'main/loss',
            ],
        ),
        trigger=log_interval,
    )
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    # -------------------------------------------------------------------------

    trainer.run()


if __name__ == '__main__':
    main()
