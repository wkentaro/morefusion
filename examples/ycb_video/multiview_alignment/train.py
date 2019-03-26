#!/usr/bin/env python

import argparse
import datetime
import pathlib
import pprint
import random
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

from contrib import Dataset
from contrib import Model


here = pathlib.Path(__file__).resolve().parent


def main():
    now = datetime.datetime.now(datetime.timezone.utc)
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
        default=[2],
        help='class ids',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
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
        default=0.0,
        help='loss scale for translation',
    )
    parser.add_argument(
        '--num-frames-scan',
        type=int,
        default=1,
        help='number of images from scan',
    )
    parser.add_argument(
        '--freeze-extractor',
        choices=['layer12', 'all'],
        default='all',
        help='freezing at',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0, help='weight decay'
    )
    parser.add_argument(
        '--loss',
        default='add',
        choices=['l1', 'add', 'add_rotation', 'add_sqrt', 'add_rotation_sqrt'],
        help='loss function',
    )
    parser.add_argument(
        '--max-epoch',
        type=int,
        default=30,
        help='max epoch',
    )
    parser.add_argument(
        '--sampling',
        type=int,
        default=15,
        help='sampling from dataset',
    )
    args = parser.parse_args()

    chainer.global_config.debug = args.debug

    termcolor.cprint('==> Started training', attrs={'bold': True})

    args.timestamp = now.isoformat()
    args.hostname = socket.gethostname()
    args.githash = objslampp.utils.githash(__file__)

    writer = tensorboardX.SummaryWriter(log_dir=args.out)
    writer_with_updater = objslampp.training.SummaryWriterWithUpdater(writer)

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
    data_train = Dataset(
        'train',
        class_ids=args.class_ids,
        num_frames_scan=args.num_frames_scan,
        sampling=args.sampling,
    )
    data_valid = Dataset(
        'val',
        class_ids=args.class_ids,
        num_frames_scan=args.num_frames_scan,
        sampling=args.sampling,
    )

    termcolor.cprint(
        'train={}, val={}'.format(len(data_train), len(data_valid)),
        attrs={'bold': True},
    )

    # model initialization
    model = Model(
        extractor=args.extractor,
        lambda_quaternion=args.lambda_quaternion,
        lambda_translation=args.lambda_translation,
        writer=writer_with_updater,
        loss_function=args.loss,
    )
    if args.gpu >= 0:
        model.to_gpu()

    # optimizer initialization
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)
    if args.weight_decay > 0:
        optimizer.add_hook(
            chainer.optimizer.WeightDecay(rate=args.weight_decay)
        )

    if args.extractor == 'resnet50':
        if args.freeze_extractor == 'all':
            for link in model.extractor.links():
                link.disable_update()
        else:
            assert args.freeze_extractor == 'layer12'
            model.extractor.conv1.disable_update()
            model.extractor.res2.disable_update()
            for link in model.extractor.links():
                if isinstance(link, chainer.links.BatchNormalization):
                    link.disable_update()
    else:
        assert args.extractor == 'vgg16'
        if args.freeze_extractor == 'all':
            for link in model.extractor.links():
                link.disable_update()
        else:
            assert args.freeze_extractor == 'layer12'
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
    writer_with_updater.setup(updater)

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out
    )

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
    evaluator = objslampp.training.extensions.PoseEstimationEvaluator(
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
    # trainer.extend(
    #     chainer.training.extensions.snapshot(
    #         filename='snapshot_iter_{.updater.iteration}.npz'
    #     ),
    #     trigger=eval_interval,
    # )
    trainer.extend(
        chainer.training.extensions.snapshot_object(
            model, filename='snapshot_model_best_auc_add.npz'
        ),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/auc/add',
            trigger=eval_interval,
        ),
    )
    trainer.extend(
        chainer.training.extensions.snapshot_object(
            model, filename='snapshot_model_best_auc_add_rotation.npz'
        ),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/auc/add_rotation',
            trigger=eval_interval,
        ),
    )

    # log
    trainer.extend(
        chainer.training.extensions.observe_lr(),
        trigger=log_interval,
    )
    trainer.extend(
        objslampp.training.extensions.LogTensorboardReport(
            writer=writer,
            trigger=log_interval,
        ),
        call_before_training=True,
    )
    trainer.extend(
        objslampp.training.extensions.ParameterTensorboardReport(
            writer=writer
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
                'main/loss',
                'main/add',
                'main/add_rotation',
                'validation/main/auc/add',
                'validation/main/auc/add_rotation',
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
                'main/loss',
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
