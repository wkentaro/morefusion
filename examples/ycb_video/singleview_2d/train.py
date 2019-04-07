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
import tensorboardX

import objslampp

import contrib


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
        '--freeze-until',
        choices=['conv4_3', 'conv3_3', 'conv2_2', 'conv1_2', 'none'],
        default='conv4_3',
        help='freeze until',
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.0001,
        help='learning rate',
    )
    parser.add_argument(
        '--max-epoch',
        type=int,
        default=30,
        help='max epoch',
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
    data_train = contrib.datasets.YCBVideoDataset('train', class_ids=[2])
    data_valid = contrib.datasets.YCBVideoDataset('val', class_ids=[2])
    class_names = objslampp.datasets.ycb_video.class_names

    termcolor.cprint('==> Dataset size', attrs={'bold': True})
    print('train={}, val={}'.format(len(data_train), len(data_valid)))

    # model initialization
    model = contrib.models.BaselineModel(
        n_fg_class=len(class_names) - 1,
        freeze_until=args.freeze_until,
    )
    if args.gpu >= 0:
        model.to_gpu()

    # optimizer initialization
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    termcolor.cprint('==> Link update rules', attrs={'bold': True})
    if args.freeze_until in ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']:
        model.extractor.conv1_1.disable_update()
        model.extractor.conv1_2.disable_update()
    if args.freeze_until in ['conv2_2', 'conv3_3', 'conv4_3']:
        model.extractor.conv2_1.disable_update()
        model.extractor.conv2_2.disable_update()
    if args.freeze_until in ['conv3_3', 'conv4_3']:
        model.extractor.conv3_1.disable_update()
        model.extractor.conv3_2.disable_update()
        model.extractor.conv3_3.disable_update()
    if args.freeze_until in ['conv4_3']:
        model.extractor.conv4_1.disable_update()
        model.extractor.conv4_2.disable_update()
        model.extractor.conv4_3.disable_update()
    for name, link in model.namedlinks():
        print(name, link.update_enabled)

    # iterator initialization
    iter_train = chainer.iterators.SerialIterator(
        data_train, batch_size=16, repeat=True, shuffle=True
    )
    iter_valid = chainer.iterators.SerialIterator(
        data_valid, batch_size=1, repeat=False, shuffle=False
    )

    updater = chainer.training.updater.StandardUpdater(
        iterator=iter_train,
        optimizer=optimizer,
        device=args.gpu,
    )
    writer_with_updater.setup(updater)

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out
    )
    trainer.extend(chainer.training.extensions.FailOnNonNumber())

    # print arguments
    msg = pprint.pformat(args.__dict__)
    msg = textwrap.indent(msg, prefix=' ' * 2)
    termcolor.cprint('==> Arguments', attrs={'bold': True})
    print(f'\n{msg}\n')

    trainer.extend(
        objslampp.training.extensions.ArgsReport(args),
        call_before_training=True,
    )

    log_interval = 10, 'iteration'
    param_log_interval = 100, 'iteration'
    eval_interval = 0.3, 'epoch'

    # evaluate
    evaluator = objslampp.training.extensions.PoseEstimationEvaluator(
        iterator=iter_valid,
        target=model,
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
        chainer.training.extensions.snapshot_object(
            model, filename='snapshot_model_best_auc_add.npz'
        ),
        trigger=chainer.training.triggers.MaxValueTrigger(
            key='validation/main/auc/add',
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
    trainer.extend(chainer.training.extensions.ProgressBar(update_interval=1))

    # -------------------------------------------------------------------------

    trainer.run()


if __name__ == '__main__':
    main()
