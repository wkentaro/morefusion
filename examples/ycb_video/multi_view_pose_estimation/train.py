#!/usr/bin/env python

import argparse
import datetime
import pathlib
import pprint
import random
import os.path as osp
import socket
import textwrap

import matplotlib.pyplot as plt
plt.switch_backend('agg')  # NOQA

import chainer
import numpy as np
import pybullet  # NOQA
import termcolor
import tensorboardX

import objslampp


here = pathlib.Path(__file__).resolve().parent


def main():
    now = datetime.datetime.now()
    default_out = str(here / 'logs' / now.strftime('%Y%m%d_%H%M%S.%f'))

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--out', default=default_out, help='output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--class-ids',
        nargs='+',
        type=int,
        default=[1],
        help='class ids',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate',
    )
    parser.add_argument(
        '--extractor',
        choices=['vgg16', 'resnet50'],
        default='vgg16',
        help='feature extractor',
    )
    parser.add_argument(
        '--lambda-quaternion',
        type=float,
        default=1.0,
        help='loss scale for quaternion',
    )
    parser.add_argument(
        '--lambda-translation',
        type=float,
        default=1.0,
        help='loss scale for translation',
    )
    parser.add_argument(
        '--num-frames-scan',
        type=int,
        default=10,
        help='number of images from scan',
    )
    args = parser.parse_args()

    chainer.global_config.debug = args.debug

    termcolor.cprint('==> Started training', attrs={'bold': True})

    args.timestamp = now.isoformat()
    args.hostname = socket.gethostname()
    args.githash = objslampp.utils.githash(__file__)

    summary_writer = tensorboardX.SummaryWriter(log_dir=args.out)

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
        'train',
        class_ids=args.class_ids,
        num_frames_scan=args.num_frames_scan,
    )
    data_valid = objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
        'val',
        class_ids=args.class_ids,
        sampling=60,
        num_frames_scan=args.num_frames_scan,
    )

    termcolor.cprint(
        'train={}, val={}'.format(len(data_train), len(data_valid)),
        attrs={'bold': True},
    )

    # model initialization
    model = objslampp.models.SimpleMV3DCNNModel(
        extractor=args.extractor,
        lambda_quaternion=args.lambda_quaternion,
        lambda_translation=args.lambda_translation,
    )
    if args.gpu >= 0:
        model.to_gpu()

    # optimizer initialization
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    if args.extractor == 'resnet50':
        model.extractor.conv1.disable_update()
        model.extractor.res2.disable_update()
        for link in model.extractor.links():
            if isinstance(link, chainer.links.BatchNormalization):
                link.disable_update()
    else:
        assert args.extractor == 'vgg16'
        model.extractor.conv1_1.disable_update()
        model.extractor.conv1_2.disable_update()
        model.extractor.conv2_1.disable_update()
        model.extractor.conv2_2.disable_update()

    # chainer.datasets.TransformDataset?

    # iterator initialization
    iter_train = chainer.iterators.SerialIterator(
        data_train, batch_size=1, repeat=True, shuffle=True
    )
    iter_valid = chainer.iterators.SerialIterator(
        data_valid, batch_size=1, repeat=False, shuffle=False
    )

    updater = chainer.training.updater.StandardUpdater(
        iterator=iter_train,
        optimizer=optimizer,
        # converter=my_converter,
        device=args.gpu,
    )

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(updater, (10, 'epoch'), out=args.out)

    # print arguments
    msg = pprint.pformat(args.__dict__)
    msg = textwrap.indent(msg, prefix=' ' * 2)
    msg = f'==> Arguments:\n\n{msg}\n'
    termcolor.cprint(msg, attrs={'bold': True})

    trainer.extend(
        objslampp.training.extensions.ArgsReport(args),
        call_before_training=True,
    )

    log_interval = 20, 'iteration'
    param_log_interval = 100, 'iteration'
    plot_interval = 20, 'iteration'
    eval_interval = 0.3, 'epoch'

    # evaluate
    evaluator = chainer.training.extensions.Evaluator(
        iterator=iter_valid,
        target=model,
        # converter=my_converter,
        device=args.gpu,
        progress_bar=True,
    )
    trainer.extend(
        evaluator,
        trigger=eval_interval,
        call_before_training=True,
    )

    # snapshot
    trainer.extend(
        chainer.training.extensions.snapshot(
            filename='snapshot_iter_{.updater.iteration}.npz'
        ),
        trigger=eval_interval,
    )
    trainer.extend(
        chainer.training.extensions.snapshot_object(
            model, filename='snapshot_model_iter_{.updater.iteration}.npz'
        ),
        trigger=eval_interval,
    )

    # log
    trainer.extend(
        chainer.training.extensions.observe_lr(),
        trigger=log_interval,
    )

    def average_child_observations(parent_keys):
        @chainer.training.make_extension(
            trigger=(1, 'iteration'),
            priority=chainer.training.extension.PRIORITY_WRITER,
        )
        def _average_child_observations(trainer):
            observation = trainer.observation
            summary = chainer.DictSummary()
            for parent in parent_keys:
                if parent in observation:
                    continue
                for name, value in observation.items():
                    if osp.dirname(name) == parent:
                        summary.add({parent: value})
            observation.update(summary.compute_mean())
        return _average_child_observations

    trainer.extend(
        average_child_observations(
            parent_keys=[
                'validation/main/loss_quaternion',
                'validation/main/loss_translation',
                'validation/main/loss',
            ],
        ),
        call_before_training=True,
    )

    trainer.extend(
        objslampp.training.extensions.LogTensorboardReport(
            writer=summary_writer,
            trigger=log_interval,
        ),
        call_before_training=True,
    )
    trainer.extend(
        objslampp.training.extensions.ParameterTensorboardReport(
            writer=summary_writer
        ),
        call_before_training=True,
        trigger=param_log_interval,
    )
    trainer.extend(
        chainer.training.extensions.PrintReport(
            [
                'epoch',
                'iteration',
                'elapsed_time',
                'lr',
                'main/loss_quaternion',
                'main/loss_translation',
                'main/loss',
                'validation/main/loss_quaternion',
                'validation/main/loss_translation',
                'validation/main/loss',
            ],
            log_report='LogTensorboardReport',
        ),
        trigger=log_interval,
        call_before_training=True,
    )
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=10))

    # plot
    assert chainer.training.extensions.PlotReport.available()
    trainer.extend(
        chainer.training.extensions.PlotReport(
            [
                'main/loss_quaternion',
                'main/loss_translation',
                'main/loss',
                'validation/main/loss_quaternion',
                'validation/main/loss_translation',
                'validation/main/loss',
            ],
            file_name='loss.png',
            trigger=plot_interval,
        ),
        trigger=(1, 'iteration'),
        call_before_training=True,
    )

    # -------------------------------------------------------------------------

    trainer.run()


if __name__ == '__main__':
    main()
