#!/usr/bin/env python

import argparse
import datetime
import pprint
import random
import socket
import textwrap

import chainer
import numpy as np
import path
import termcolor
import tensorboardX

import objslampp

import contrib


here = path.Path(__file__).abspath().parent


def concat_list_of_examples(list_of_examples, device=None, padding=None):
    batch = []
    for examples in list_of_examples:
        batch.extend(examples)
    return chainer.dataset.concat_examples(
        batch, device=device, padding=padding
    )


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
        '--dataset',
        choices=['ycb_video'],
        default='ycb_video',
        help='dataset',
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
    parser.add_argument(
        '--call-evaluation-before-training',
        action='store_true',
        help='call evaluation before training',
    )
    parser.add_argument(
        '--class-ids',
        type=int,
        nargs='*',
        default=objslampp.datasets.ycb_video.class_ids_asymmetric,
        help='class ids',
    )
    parser.add_argument(
        '--num-syn',
        type=float,
        default=0.25,
        help='number of synthetic examples used',
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
    data_train = contrib.datasets.YCBVideoDataset(
        'train', class_ids=args.class_ids, num_syn=args.num_syn
    )
    data_valid = contrib.datasets.YCBVideoDataset(
        'val', class_ids=args.class_ids
    )
    class_names = objslampp.datasets.ycb_video.class_names

    termcolor.cprint('==> Dataset size', attrs={'bold': True})
    print('train={}, val={}'.format(len(data_train), len(data_valid)))

    # model initialization
    model = contrib.models.BaselineModel(
        n_fg_class=len(class_names) - 1,
    )
    if args.gpu >= 0:
        model.to_gpu()

    # optimizer initialization
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    optimizer.setup(model)

    termcolor.cprint('==> Link update rules', attrs={'bold': True})
    for name, link in model.namedlinks():
        print(name, link.update_enabled)

    # iterator initialization
    iter_train = contrib.iterators.MultiExamplePerImageSerialIterator(
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

    log_interval = 1, 'iteration'
    param_log_interval = 100, 'iteration'
    eval_interval = 0.3, 'epoch'

    # evaluate
    evaluator = objslampp.training.extensions.PoseEstimationEvaluator(
        iterator=iter_valid,
        converter=concat_list_of_examples,
        target=model,
        device=args.gpu,
        progress_bar=True,
    )
    trainer.extend(
        evaluator,
        trigger=eval_interval,
        call_before_training=args.call_evaluation_before_training,
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
                'main/add_s',
                'validation/main/auc/add',
                'validation/main/auc/add_s',
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
