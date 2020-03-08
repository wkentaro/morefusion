#!/usr/bin/env python

import argparse
import datetime
import os.path as osp
import pprint
import random
import socket
import textwrap

import chainer
from chainer.training import extensions as E
import numpy as np
import path
import tensorboardX
import termcolor

import morefusion

import contrib


home = path.Path("~").expanduser()
here = path.Path(__file__).abspath().parent


def transform(in_data):
    assert in_data["class_id"].dtype == np.int32
    assert in_data["rgb"].dtype == np.uint8
    assert in_data["rgb"].shape == (256, 256, 3)

    assert in_data["pcd"].dtype == np.float64
    assert in_data["pcd"].shape == (256, 256, 3)
    in_data["pcd"] = in_data["pcd"].astype(np.float32)

    assert in_data["quaternion_true"].dtype == np.float64
    in_data["quaternion_true"] = in_data["quaternion_true"].astype(np.float32)

    assert in_data["translation_true"].dtype == np.float64
    in_data["translation_true"] = in_data["translation_true"].astype(
        np.float32
    )

    in_data.pop("pitch")
    in_data.pop("origin")

    in_data.pop("grid_target")
    in_data.pop("grid_nontarget")
    in_data.pop("grid_empty")

    in_data.pop("grid_target_full")
    in_data.pop("grid_nontarget_full")
    return in_data


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--multi-node", action="store_true", help="multi node")
    parser.add_argument("--out", help="output directory")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--gpu", type=int, default=0, help="gpu id")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--lr", type=float, default=0.0001, help="learning rate",
    )
    parser.add_argument(
        "--max-epoch", type=int, default=30, help="max epoch",
    )
    parser.add_argument(
        "--call-evaluation-before-training",
        action="store_true",
        help="call evaluation before training",
    )

    def argparse_type_class_ids(string):
        if string == "all":
            n_class = len(morefusion.datasets.ycb_video.class_names)
            class_ids = np.arange(n_class)[1:].tolist()
        elif string == "asymmetric":
            class_ids = (
                morefusion.datasets.ycb_video.class_ids_asymmetric.tolist()
            )
        elif string == "symmetric":
            class_ids = (
                morefusion.datasets.ycb_video.class_ids_symmetric.tolist()
            )
        else:
            class_ids = [int(x) for x in string.split(",")]
        return class_ids

    parser.add_argument(
        "--class-ids",
        type=argparse_type_class_ids,
        default="all",
        help="class id (e.g., 'all', 'asymmetric', 'symmetric', '1,6,9')",
    )
    parser.add_argument(
        "--pretrained-model", help="pretrained model",
    )
    parser.add_argument(
        "--note", help="note",
    )
    parser.add_argument(
        "--pretrained-resnet18",
        action="store_true",
        help="pretrained resnet18",
    )
    parser.add_argument(
        "--centerize-pcd", action="store_true", help="centerize pcd",
    )
    parser.add_argument(
        "--resume", help="resume",
    )
    parser.add_argument(
        "--loss",
        choices=["add/add_s", "add->add/add_s|1"],
        default="add->add/add_s|1",
        help="loss",
    )
    args = parser.parse_args()

    chainer.global_config.debug = args.debug

    # -------------------------------------------------------------------------

    # device initialization
    if args.multi_node:
        import chainermn

        comm = chainermn.create_communicator("pure_nccl")
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

        termcolor.cprint("==> Started training", attrs={"bold": True})

    if args.out is None:
        if not args.multi_node or comm.rank == 0:
            args.out = osp.join(here, "logs", now.strftime("%Y%m%d_%H%M%S.%f"))
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
        termcolor.cprint("==> Dataset size", attrs={"bold": True})

        data_ycb_trainreal = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            "trainreal", class_ids=args.class_ids, augmentation=True
        )
        data_ycb_syn = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            "syn", class_ids=args.class_ids, augmentation=True
        )
        data_ycb_syn = morefusion.datasets.RandomSamplingDataset(
            data_ycb_syn, len(data_ycb_trainreal)
        )
        data_my_train = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(  # NOQA
            "train", class_ids=args.class_ids, augmentation=True
        )
        data_train = chainer.datasets.ConcatenatedDataset(
            data_ycb_trainreal, data_ycb_syn, data_my_train
        )
        print(
            f"ycb_trainreal={len(data_ycb_trainreal)}, "
            f"ycb_syn={len(data_ycb_syn)}, my_train={len(data_my_train)}"
        )
        del data_ycb_trainreal, data_ycb_syn, data_my_train

        data_ycb_val = morefusion.datasets.YCBVideoRGBDPoseEstimationDatasetReIndexed(  # NOQA
            "val", class_ids=args.class_ids
        )
        data_my_val = morefusion.datasets.MySyntheticYCB20190916RGBDPoseEstimationDatasetReIndexed(  # NOQA
            "val", class_ids=args.class_ids
        )
        data_valid = chainer.datasets.ConcatenatedDataset(
            data_ycb_val, data_my_val,
        )
        print(f"ycb_val={len(data_ycb_val)}, my_val={len(data_my_val)}")
        del data_ycb_val, data_my_val

        data_train = chainer.datasets.TransformDataset(data_train, transform)
        data_valid = chainer.datasets.TransformDataset(data_valid, transform)

    if args.multi_node:
        data_train = chainermn.scatter_dataset(
            data_train, comm, shuffle=True, seed=args.seed
        )
        data_valid = chainermn.scatter_dataset(
            data_valid, comm, shuffle=False, seed=args.seed
        )

    args.class_names = morefusion.datasets.ycb_video.class_names.tolist()

    loss = args.loss
    if loss == "add->add/add_s|1":
        loss = "add"

    # model initialization
    model = contrib.models.Model(
        n_fg_class=len(args.class_names) - 1,
        centerize_pcd=args.centerize_pcd,
        pretrained_resnet18=args.pretrained_resnet18,
        loss=loss,
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
        termcolor.cprint("==> Link update rules", attrs={"bold": True})
        for name, link in model.namedlinks():
            print(name, link.update_enabled)

    # iterator initialization
    iter_train = chainer.iterators.MultiprocessIterator(
        data_train, batch_size=16 // n_gpu, repeat=True, shuffle=True,
    )
    iter_valid = chainer.iterators.MultiprocessIterator(
        data_valid, batch_size=16, repeat=False, shuffle=False,
    )

    updater = chainer.training.StandardUpdater(
        iterator=iter_train, optimizer=optimizer, device=device,
    )
    if not args.multi_node or comm.rank == 0:
        writer = tensorboardX.SummaryWriter(log_dir=args.out)
        writer_with_updater = morefusion.training.SummaryWriterWithUpdater(
            writer
        )
        writer_with_updater.setup(updater)

    # -------------------------------------------------------------------------

    trainer = chainer.training.Trainer(
        updater, (args.max_epoch, "epoch"), out=args.out
    )
    trainer.extend(E.FailOnNonNumber())

    @chainer.training.make_extension(trigger=(1, "iteration"))
    def update_loss(trainer):
        updater = trainer.updater
        optimizer = updater.get_optimizer("main")
        target = optimizer.target
        assert trainer.stop_trigger.unit == "epoch"

        if args.loss == "add->add/add_s|1":
            if updater.epoch_detail < 1:
                assert target._loss == "add"
            else:
                target._loss = "add/add_s"
        else:
            assert args.loss in ["add/add_s"]
            return

    trainer.extend(update_loss)

    log_interval = 10, "iteration"
    eval_interval = 0.25, "epoch"

    # evaluate
    evaluator = morefusion.training.extensions.PoseEstimationEvaluator(
        iterator=iter_valid, target=model, device=device, progress_bar=True,
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
        msg = textwrap.indent(msg, prefix=" " * 2)
        termcolor.cprint("==> Arguments", attrs={"bold": True})
        print(f"\n{msg}\n")

        trainer.extend(
            morefusion.training.extensions.ArgsReport(args),
            call_before_training=True,
        )

        # snapshot
        trigger_best_add = chainer.training.triggers.MinValueTrigger(
            key="validation/main/add_or_add_s", trigger=eval_interval,
        )
        trigger_best_auc = chainer.training.triggers.MaxValueTrigger(
            key="validation/main/auc/add_or_add_s", trigger=eval_interval,
        )
        trainer.extend(
            E.snapshot(filename="snapshot_trainer_latest.npz"),
            trigger=eval_interval,
        )
        trainer.extend(
            E.snapshot_object(model, filename="snapshot_model_latest.npz"),
            trigger=eval_interval,
        )
        trainer.extend(
            E.snapshot_object(model, filename="snapshot_model_best_add.npz"),
            trigger=trigger_best_add,
        )
        trainer.extend(
            E.snapshot_object(model, filename="snapshot_model_best_auc.npz"),
            trigger=trigger_best_auc,
        )

        # log
        trainer.extend(
            morefusion.training.extensions.LogTensorboardReport(
                writer=writer, trigger=log_interval,
            ),
            call_before_training=True,
        )
        trainer.extend(
            E.PrintReport(
                [
                    "epoch",
                    "iteration",
                    "elapsed_time",
                    "main/loss",
                    "main/add_or_add_s",
                    "validation/main/auc/add_or_add_s",
                ],
                log_report="LogTensorboardReport",
            ),
            trigger=log_interval,
            call_before_training=True,
        )
        trainer.extend(E.ProgressBar(update_interval=1))

    # -------------------------------------------------------------------------

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == "__main__":
    main()
