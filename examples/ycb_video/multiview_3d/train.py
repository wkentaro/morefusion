#!/usr/bin/env python

import argparse
import datetime
import os.path as osp
import pprint
import random
import re
import socket
import textwrap

import chainer
from chainer.training import extensions as E
import numpy as np
import path
import termcolor
import tensorboardX
import yaml

import objslampp

import contrib


home = path.Path('~').expanduser()
here = path.Path(__file__).abspath().parent


def concat_list_of_examples(list_of_examples, device=None, padding=None):
    batch = []
    for examples in list_of_examples:
        batch.extend(examples)
    return chainer.dataset.concat_examples(
        batch, device=device, padding=padding
    )


def sigmoid(x):
    return 1 / (1 + np.exp(- x))


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--multi-node', action='store_true', help='multi node'
    )
    parser.add_argument('--out', help='output directory')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--dataset',
        choices=['ycb_video', 'my_synthetic'],
        default='ycb_video',
        help='dataset',
    )
    parser.add_argument(
        '--augmentation',
        nargs='*',
        default=None,
        choices=['rgb', 'depth'],
        help='augmentation',
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
        default=objslampp.datasets.ycb_video.class_ids_asymmetric.tolist(),
        help='class id',
    )
    parser.add_argument(
        '--use-occupancy',
        action='store_true',
        help='use occupancy',
    )

    def argparse_type_loss(string):
        patterns = [
            'add',
            r'add\+occupancy',
            'add_s',
            r'add_s\+occupancy',
            'add/add_s',
            r'add/add_s\+occupancy',
            r'add\+add_s',
            r'add\+add_s\|linear',
            r'add\+add_s\|sigmoid',
            r'add\+add_s\|step\|\d+'
            'overlap',
            r'overlap\+occupancy',
            'iou',
            r'iou\+occupancy',
        ]
        for pattern in patterns:
            if re.match(pattern, string):
                break
        else:
            raise argparse.ArgumentTypeError
        return string

    parser.add_argument(
        '--loss',
        default='add/add_s',
        type=argparse_type_loss,
        help='loss',
    )
    parser.add_argument(
        '--loss-scale',
        type=yaml.safe_load,
        help='loss scale e.g., {occupancy: 1.0}',
    )
    parser.add_argument(
        '--num-syn',
        type=float,
        default=0.25,
        help='number of synthetic examples used',
    )
    parser.add_argument(
        '--pretrained-model',
        help='pretrained model',
    )
    args = parser.parse_args()

    chainer.global_config.debug = args.debug

    # -------------------------------------------------------------------------

    # device initialization
    if args.multi_node:
        import chainermn

        comm = chainermn.create_communicator('hierarchical')
        device = comm.intra_rank
        n_gpu = comm.size
    else:
        device = args.gpu
        n_gpu = 1

    if not args.multi_node or comm.rank == 0:
        now = datetime.datetime.now(datetime.timezone.utc)
        args.timestamp = now.isoformat()
        args.hostname = socket.gethostname()
        args.githash = objslampp.utils.githash(__file__)

        termcolor.cprint('==> Started training', attrs={'bold': True})

    if args.out is None:
        if not args.multi_node or comm.rank == 0:
            args.out = osp.join(here, 'logs', now.strftime('%Y%m%d_%H%M%S.%f'))
        else:
            args.out = None
        if args.multi_node:
            args.out = comm.bcast_obj(args.out)

    if device >= 0:
        chainer.cuda.get_device_from_id(device).use()

    # seed initialization
    random.seed(args.seed)
    np.random.seed(args.seed)
    if device >= 0:
        chainer.cuda.cupy.random.seed(args.seed)

    # dataset initialization
    return_occupancy_grids = \
        args.use_occupancy or args.loss == 'add/add_s+complete'
    data_train = None
    data_valid = None
    if not args.multi_node or comm.rank == 0:
        if args.dataset == 'ycb_video':
            data_train = contrib.datasets.YCBVideoDataset(
                'train',
                class_ids=args.class_ids,
                # augmentation=args.augmentation,
                # return_occupancy_grids=return_occupancy_grids,
                num_syn=args.num_syn,
            )
        elif args.dataset == 'my_synthetic':
            root_dir = home / 'data/datasets/wkentaro/objslampp/ycb_video/synthetic_data/20190715_113906.827534'  # NOQA
            data = contrib.datasets.MySyntheticDataset(
                root_dir=root_dir,
                class_ids=args.class_ids,
                augmentation=args.augmentation,
                return_occupancy_grids=return_occupancy_grids,
            )
            assert len(data.root_dir.dirs()) == 750
            assert len(data) == 750 * 15
            data_train, data_valid = chainer.datasets.split_dataset(
                data, split_at=600 * 15
            )
        else:
            raise ValueError(f'unsupported dataset: {args.dataset}')

        if data_valid is None:
            data_valid = contrib.datasets.YCBVideoDataset(
                'val',
                class_ids=args.class_ids,
                # return_occupancy_grids=return_occupancy_grids,
            )

        termcolor.cprint('==> Dataset size', attrs={'bold': True})
        print(f'train={len(data_train)}, val={len(data_valid)}')

    if args.multi_node:
        data_train = chainermn.scatter_dataset(data_train, comm, shuffle=True)

    args.class_names = objslampp.datasets.ycb_video.class_names.tolist()

    loss = args.loss
    if re.match(r'add\+add_s\|.*', loss):
        loss = 'add+add_s'

    # model initialization
    model = contrib.models.BaselineModel(
        n_fg_class=len(args.class_names[1:]),
        # use_occupancy=args.use_occupancy,
        loss=loss,
        loss_scale=args.loss_scale,
    )
    if args.pretrained_model is not None:
        chainer.serializers.load_npz(args.pretrained_model, model)
    if device >= 0:
        model.to_gpu()

    # optimizer initialization
    optimizer = chainer.optimizers.Adam(alpha=args.lr)
    if args.multi_node:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)

    if not args.multi_node or comm.rank == 0:
        termcolor.cprint('==> Link update rules', attrs={'bold': True})
        for name, link in model.namedlinks():
            print(name, link.update_enabled)

    # iterator initialization
    iter_train = contrib.iterators.MultiExamplePerImageSerialIterator(
        data_train, batch_size=16 // n_gpu, repeat=True, shuffle=True
    )
    iter_valid = chainer.iterators.SerialIterator(
        data_valid, batch_size=1, repeat=False, shuffle=False
    )

    updater = chainer.training.StandardUpdater(
        iterator=iter_train,
        optimizer=optimizer,
        device=device,
    )
    if not args.multi_node or comm.rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=args.out)
        writer_with_updater = objslampp.training.SummaryWriterWithUpdater(
            writer
        )
        writer_with_updater.setup(updater)

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out
    )
    trainer.extend(E.FailOnNonNumber())

    @chainer.training.make_extension(trigger=(1, 'iteration'))
    def update_loss_scale(trainer):
        updater = trainer.updater
        optimizer = updater.get_optimizer('main')
        target = optimizer.target
        assert trainer.stop_trigger.unit == 'epoch'
        max_epoch = trainer.stop_trigger.period

        if args.loss == 'add+add_s|linear':
            loss_scale_add = 1 - updater.epoch_detail / max_epoch
        elif args.loss == 'add+add_s|sigmoid':
            loss_scale_add = 1 - sigmoid(updater.epoch_detail - max_epoch / 2)
        elif re.match(r'add\+add_s\|step\|\d+', args.loss):
            match = re.match(r'add\+add_s\|step\|(\d+)', args.loss)
            epoch_anchor = int(match.groups()[0])
            loss_scale_add = 1
            if updater.epoch_detail > epoch_anchor:
                loss_scale_add = 0
        else:
            return
        target._loss_scale['add+add_s'] = loss_scale_add

        report = {f'loss_scale/{k}': v for k, v in target._loss_scale.items()}
        chainer.report(report)

    trainer.extend(update_loss_scale)

    if not args.multi_node or comm.rank == 0:
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
            device=device,
            progress_bar=True,
        )
        trainer.extend(
            evaluator,
            trigger=eval_interval,
            call_before_training=args.call_evaluation_before_training,
        )

        # snapshot
        trigger_best_add = chainer.training.triggers.MaxValueTrigger(
            key='validation/main/auc/add',
            trigger=eval_interval,
        )
        trigger_best_add_s = chainer.training.triggers.MaxValueTrigger(
            key='validation/main/auc/add_s',
            trigger=eval_interval,
        )
        trainer.extend(
            E.snapshot(filename='snapshot_trainer_latest.npz'),
            trigger=eval_interval,
        )
        trainer.extend(
            E.snapshot_object(
                model, filename='snapshot_model_latest.npz'
            ),
            trigger=eval_interval,
        )
        trainer.extend(
            E.snapshot_object(
                model, filename='snapshot_model_best_auc_add.npz'
            ),
            trigger=trigger_best_add,
        )
        trainer.extend(
            E.snapshot_object(
                model, filename='snapshot_model_best_auc_add_s.npz'
            ),
            trigger=trigger_best_add_s,
        )

        # log
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
            E.PrintReport(
                [
                    'epoch',
                    'iteration',
                    'elapsed_time',
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
        trainer.extend(
            E.ProgressBar(update_interval=1)
        )

    # -------------------------------------------------------------------------

    trainer.run()


if __name__ == '__main__':
    main()
