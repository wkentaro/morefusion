#!/usr/bin/env python

import argparse
import datetime
import json
import os.path as osp
import pprint
import random
import socket
import textwrap

import chainer
from chainer.training import extensions as E
import numpy as np
import path
import termcolor
import tensorboardX

import morefusion

import contrib


home = path.Path('~').expanduser()
here = path.Path(__file__).abspath().parent


class Transform:

    def __init__(self, train, with_occupancy):
        assert train in [True, False]
        assert with_occupancy in [True, False]
        self._train = train
        self._with_occupancy = with_occupancy
        self._random_state = np.random.mtrand._rand

    def __call__(self, in_data):
        assert in_data['class_id'].dtype == np.int32
        assert in_data['rgb'].dtype == np.uint8
        in_data['pcd'] = in_data['pcd'].astype(np.float32)
        in_data['quaternion_true'] = in_data['quaternion_true'].astype(
            np.float32
        )
        in_data['translation_true'] = in_data['translation_true'].astype(
            np.float32
        )

        if self._with_occupancy:
            in_data['origin'] = in_data['origin'].astype(np.float32)
            in_data['pitch'] = in_data['pitch'].astype(np.float32)

            grid_target = in_data.pop('grid_target') > 0.5
            grid_nontarget = in_data.pop('grid_nontarget') > 0.5
            grid_empty = in_data.pop('grid_empty') > 0.5
            grid_nontarget = grid_nontarget ^ grid_target
            grid_empty = grid_empty ^ grid_target

            grid_target_full = in_data.pop('grid_target_full')
            assert np.isin(grid_target_full, [0, 1]).all()
            grid_target_full = grid_target_full.astype(bool)

            grid_nontarget_full = in_data.pop('grid_nontarget_full')
            nontarget_ids = np.unique(grid_nontarget_full)
            nontarget_ids = nontarget_ids[nontarget_ids > 0]
            if len(nontarget_ids) > 0:
                if len(nontarget_ids) > 1:
                    nontarget_ids = self._random_state.choice(
                        nontarget_ids,
                        size=self._random_state.randint(
                            1, len(nontarget_ids) + 1
                        ),
                        replace=False,
                    )
                grid_nontarget_full = np.isin(
                    grid_nontarget_full, nontarget_ids
                )
            else:
                grid_nontarget_full = np.zeros_like(grid_target)
            grid_nontarget_full = grid_nontarget_full ^ grid_target_full

            if self._train:
                cases = [
                    'none',
                    'empty',
                    'nontarget',
                    'empty+nontarget',
                    'nontarget_full',
                    'empty+nontarget_full',
                    'other_full',
                    'nontarget_full+other_full',
                    'empty+nontarget_full+other_full',
                ]
                case = self._random_state.choice(cases)
            else:
                case = 'empty+nontarget'

            if case == 'none':
                grid_nontarget_empty = np.zeros_like(grid_target)
            elif case == 'empty+nontarget_full+other_full':
                grid_nontarget_empty = ~grid_target_full
            else:
                if case == 'empty':
                    grid_nontarget_empty = grid_empty
                elif case == 'nontarget':
                    grid_nontarget_empty = grid_nontarget
                elif case == 'empty+nontarget':
                    grid_nontarget_empty = grid_nontarget | grid_empty
                elif case == 'nontarget_full':
                    grid_nontarget_empty = grid_nontarget_full
                elif case == 'empty+nontarget_full':
                    grid_nontarget_empty = grid_empty | grid_nontarget_full
                else:
                    grid_other_full = (
                        ~grid_target_full &
                        ~grid_nontarget_full &
                        ~grid_empty &
                        ~grid_target &
                        ~grid_nontarget
                    )
                    if case == 'other_full':
                        grid_nontarget_empty = grid_other_full
                    else:
                        assert case == 'nontarget_full+other_full'
                        grid_nontarget_empty = \
                            grid_nontarget_full | grid_other_full

            in_data['grid_target'] = grid_target
            in_data['grid_nontarget_empty'] = grid_nontarget_empty
            assert in_data['grid_target'].dtype == bool
            assert in_data['grid_nontarget_empty'].dtype == bool
        else:
            in_data.pop('pitch')
            in_data.pop('origin')

            in_data.pop('grid_target')
            in_data.pop('grid_nontarget')
            in_data.pop('grid_empty')

            in_data.pop('grid_target_full')
            in_data.pop('grid_nontarget_full')
        return in_data


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

    def argparse_type_class_ids(string):
        if string == 'all':
            n_class = len(morefusion.datasets.ycb_video.class_names)
            class_ids = np.arange(n_class)[1:].tolist()
        elif string == 'asymmetric':
            class_ids = morefusion.datasets.ycb_video.class_ids_asymmetric\
                .tolist()
        elif string == 'symmetric':
            class_ids = morefusion.datasets.ycb_video.class_ids_symmetric\
                .tolist()
        else:
            class_ids = [int(x) for x in string.split(',')]
        return class_ids

    parser.add_argument(
        '--class-ids',
        type=argparse_type_class_ids,
        default='all',
        help="class id (e.g., 'all', 'asymmetric', 'symmetric', '1,6,9')",
    )
    parser.add_argument(
        '--pretrained-model',
        help='pretrained model',
    )
    parser.add_argument(
        '--with-occupancy',
        action='store_true',
        help='with occupancy',
    )
    parser.add_argument(
        '--note',
        help='note',
    )
    parser.add_argument(
        '--pretrained-resnet18',
        action='store_true',
        help='pretrained resnet18',
    )
    parser.add_argument(
        '--resume',
        help='resume',
    )
    parser.add_argument(
        '--loss',
        choices=[
            'add/add_s',
            'add/add_s+occupancy',
            'add->add/add_s|1',
            'add->add/add_s|1+occupancy',
        ],
        default='add->add/add_s|1',
        help='loss',
    )
    parser.add_argument(
        '--loss-scale',
        type=json.loads,
        default={
            'occupancy': 1.0,
        },
        help='loss scale',
    )
    args = parser.parse_args()

    chainer.global_config.debug = args.debug

    # -------------------------------------------------------------------------

    # device initialization
    if args.multi_node:
        import chainermn

        comm = chainermn.create_communicator('pure_nccl')
        device = comm.intra_rank
        n_gpu = comm.size
    else:
        device = args.gpu
        n_gpu = 1

    if not args.multi_node or comm.rank == 0:
        now = datetime.datetime.now(datetime.timezone.utc)
        args.timestamp = now.isoformat()
        args.hostname = socket.gethostname()
        args.githash = morefusion.utils.githash(__file__)

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
    data_train = None
    data_valid = None
    if not args.multi_node or comm.rank == 0:
        termcolor.cprint('==> Dataset size', attrs={'bold': True})

        data_ycb_trainreal = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            'trainreal', class_ids=args.class_ids, augmentation=True
        )
        data_ycb_syn = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            'syn', class_ids=args.class_ids, augmentation=True
        )
        data_ycb_syn = morefusion.datasets.RandomSamplingDataset(
            data_ycb_syn, len(data_ycb_trainreal)
        )
        data_my_train = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(  # NOQA
            'train', class_ids=args.class_ids, augmentation=True
        )
        data_train = chainer.datasets.ConcatenatedDataset(
            data_ycb_trainreal, data_ycb_syn, data_my_train
        )
        print(f'ycb_trainreal={len(data_ycb_trainreal)}, '
              f'ycb_syn={len(data_ycb_syn)}, my_train={len(data_my_train)}')
        del data_ycb_trainreal, data_ycb_syn, data_my_train

        data_ycb_val = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            'val', class_ids=args.class_ids
        )
        data_my_val = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(  # NOQA
            'val', class_ids=args.class_ids
        )
        data_valid = chainer.datasets.ConcatenatedDataset(
            data_ycb_val, data_my_val,
        )
        print(f'ycb_val={len(data_ycb_val)}, my_val={len(data_my_val)}')
        del data_ycb_val, data_my_val

        data_train = chainer.datasets.TransformDataset(
            data_train,
            Transform(train=True, with_occupancy=args.with_occupancy),
        )
        data_valid = chainer.datasets.TransformDataset(
            data_valid,
            Transform(train=False, with_occupancy=args.with_occupancy),
        )
        print(f'train={len(data_train)}, valid={len(data_valid)}')

    if args.multi_node:
        data_train = chainermn.scatter_dataset(
            data_train, comm, shuffle=True, seed=args.seed
        )
        data_valid = chainermn.scatter_dataset(
            data_valid, comm, shuffle=False, seed=args.seed
        )

    args.class_names = morefusion.datasets.ycb_video.class_names.tolist()

    loss = args.loss
    if loss == 'add->add/add_s|1':
        loss = 'add'
    elif loss == 'add->add/add_s|1+occupancy':
        loss = 'add+occupancy'

    # model initialization
    model = contrib.models.Model(
        n_fg_class=len(args.class_names[1:]),
        pretrained_resnet18=args.pretrained_resnet18,
        with_occupancy=args.with_occupancy,
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

    if args.pretrained_resnet18:
        model.resnet_extractor.init_block.disable_update()
        model.resnet_extractor.res2.disable_update()
        for link in model.links():
            if isinstance(link, chainer.links.BatchNormalization):
                link.disable_update()

    if not args.multi_node or comm.rank == 0:
        termcolor.cprint('==> Link update rules', attrs={'bold': True})
        for name, link in model.namedlinks():
            print(name, link.update_enabled)

    # iterator initialization
    iter_train = chainer.iterators.MultithreadIterator(
        data_train,
        batch_size=16 // n_gpu,
        repeat=True,
        shuffle=True,
    )
    iter_valid = chainer.iterators.MultithreadIterator(
        data_valid,
        batch_size=48,
        repeat=False,
        shuffle=False,
    )

    updater = chainer.training.StandardUpdater(
        iterator=iter_train,
        optimizer=optimizer,
        device=device,
    )
    if not args.multi_node or comm.rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=args.out)
        writer_with_updater = morefusion.training.SummaryWriterWithUpdater(
            writer
        )
        writer_with_updater.setup(updater)

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(
        updater, (args.max_epoch, 'epoch'), out=args.out
    )
    trainer.extend(E.FailOnNonNumber())

    @chainer.training.make_extension(trigger=(1, 'iteration'))
    def update_loss(trainer):
        updater = trainer.updater
        optimizer = updater.get_optimizer('main')
        target = optimizer.target
        assert trainer.stop_trigger.unit == 'epoch'

        if args.loss == 'add->add/add_s|1':
            if updater.epoch_detail < 1:
                assert target._loss == 'add'
            else:
                target._loss = 'add/add_s'
        elif args.loss == 'add->add/add_s|1+occupancy':
            if updater.epoch_detail < 1:
                assert target._loss == 'add+occupancy'
            else:
                target._loss = 'add/add_s+occupancy'
        else:
            assert args.loss in ['add/add_s', 'add/add_s+occupancy']
            return

    trainer.extend(update_loss)

    log_interval = 10, 'iteration'
    eval_interval = 0.25, 'epoch'

    # evaluate
    evaluator = morefusion.training.extensions.PoseEstimationEvaluator(
        iterator=iter_valid,
        target=model,
        device=device,
        progress_bar=True,
    )
    if args.multi_node:
        evaluator.comm = comm
    trainer.extend(
        evaluator,
        trigger=eval_interval,
        call_before_training=args.call_evaluation_before_training,
    )

    if not args.multi_node or comm.rank == 0:
        # print arguments
        msg = pprint.pformat(args.__dict__)
        msg = textwrap.indent(msg, prefix=' ' * 2)
        termcolor.cprint('==> Arguments', attrs={'bold': True})
        print(f'\n{msg}\n')

        trainer.extend(
            morefusion.training.extensions.ArgsReport(args),
            call_before_training=True,
        )

        # snapshot
        trigger_best_add = chainer.training.triggers.MinValueTrigger(
            key='validation/main/add_or_add_s',
            trigger=eval_interval,
        )
        trigger_best_auc = chainer.training.triggers.MaxValueTrigger(
            key='validation/main/auc/add_or_add_s',
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
                model, filename='snapshot_model_best_add.npz'
            ),
            trigger=trigger_best_add,
        )
        trainer.extend(
            E.snapshot_object(
                model, filename='snapshot_model_best_auc.npz'
            ),
            trigger=trigger_best_auc,
        )

        # log
        trainer.extend(
            morefusion.training.extensions.LogTensorboardReport(
                writer=writer,
                trigger=log_interval,
            ),
            call_before_training=True,
        )
        trainer.extend(
            E.PrintReport(
                [
                    'epoch',
                    'iteration',
                    'elapsed_time',
                    'main/loss',
                    'main/add_or_add_s',
                    'validation/main/auc/add_or_add_s',
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

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
