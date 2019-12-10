from __future__ import division

import argparse
import multiprocessing
import numpy as np

import chainer
from chainer.datasets import TransformDataset
import chainer.functions as F
import chainer.links as L
from chainer.optimizer_hooks import WeightDecay
from chainer import serializers
from chainer import training
from chainer.training import extensions

import chainermn

from chainercv.chainer_experimental.training.extensions import make_shift

from chainercv.extensions import InstanceSegmentationCOCOEvaluator
from chainercv.links import MaskRCNNFPNResNet50

from chainercv.links.model.fpn import bbox_head_loss_post
from chainercv.links.model.fpn import bbox_head_loss_pre
from chainercv.links.model.fpn import mask_head_loss_post
from chainercv.links.model.fpn import mask_head_loss_pre
from chainercv.links.model.fpn import rpn_loss

import morefusion

from transforms import Affine
from transforms import AsType
from transforms import ClassIds2FGClassIds
from transforms import Compose
from transforms import Dict2Tuple
from transforms import HWC2CHW
from transforms import MaskRCNNTransform
from transforms import RGBAugmentation


# https://docs.chainer.org/en/stable/tips.html#my-training-process-gets-stuck-when-using-multiprocessiterator
try:
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


chainer.config.cv_resize_backend = 'cv2'


class TrainChain(chainer.Chain):

    def __init__(self, model):
        super(TrainChain, self).__init__()
        with self.init_scope():
            self.model = model

    def forward(self, imgs, bboxes, labels, masks=None):
        if np.any([len(bbox) == 0 for bbox in bboxes]):
            return chainer.Variable(self.xp.zeros((), dtype=np.float32))

        B = len(imgs)
        pad_size = np.array(
            [im.shape[1:] for im in imgs]).max(axis=0)
        pad_size = (
            np.ceil(
                pad_size / self.model.stride) * self.model.stride).astype(int)
        x = np.zeros(
            (len(imgs), 3, pad_size[0], pad_size[1]), dtype=np.float32)
        for i, img in enumerate(imgs):
            _, H, W = img.shape
            x[i, :, :H, :W] = img
        x = self.xp.array(x)

        bboxes = [self.xp.array(bbox) for bbox in bboxes]
        labels = [self.xp.array(label) for label in labels]
        sizes = [img.shape[1:] for img in imgs]

        with chainer.using_config('train', False):
            hs = self.model.extractor(x)

        rpn_locs, rpn_confs = self.model.rpn(hs)
        anchors = self.model.rpn.anchors(h.shape[2:] for h in hs)
        rpn_loc_loss, rpn_conf_loss = rpn_loss(
            rpn_locs, rpn_confs, anchors, sizes, bboxes)

        rois, roi_indices = self.model.rpn.decode(
            rpn_locs, rpn_confs, anchors, x.shape)
        rois = self.xp.vstack([rois] + bboxes)
        roi_indices = self.xp.hstack(
            [roi_indices]
            + [self.xp.array((i,) * len(bbox))
               for i, bbox in enumerate(bboxes)])
        rois, roi_indices = self.model.bbox_head.distribute(rois, roi_indices)
        rois, roi_indices, head_gt_locs, head_gt_labels = bbox_head_loss_pre(
            rois, roi_indices, self.model.bbox_head.std, bboxes, labels)
        head_locs, head_confs = self.model.bbox_head(hs, rois, roi_indices)
        head_loc_loss, head_conf_loss = bbox_head_loss_post(
            head_locs, head_confs,
            roi_indices, head_gt_locs, head_gt_labels, B)

        mask_loss = 0
        if masks is not None:
            # For reducing unnecessary CPU/GPU copy, `masks` is kept in CPU.
            pad_masks = [
                np.zeros(
                    (mask.shape[0], pad_size[0], pad_size[1]), dtype=np.bool)
                for mask in masks]
            for i, mask in enumerate(masks):
                _, H, W = mask.shape
                pad_masks[i][:, :H, :W] = mask
            masks = pad_masks

            mask_rois, mask_roi_indices, gt_segms, gt_mask_labels =\
                mask_head_loss_pre(
                    rois, roi_indices, masks, bboxes,
                    head_gt_labels, self.model.mask_head.segm_size)
            n_roi = sum([len(roi) for roi in mask_rois])
            if n_roi > 0:
                segms = self.model.mask_head(hs, mask_rois, mask_roi_indices)
                mask_loss = mask_head_loss_post(
                    segms, mask_roi_indices, gt_segms, gt_mask_labels, B)
            else:
                # Compute dummy variables to complete the computational graph
                mask_rois[0] = self.xp.array([[0, 0, 1, 1]], dtype=np.float32)
                mask_roi_indices[0] = self.xp.array([0], dtype=np.int32)
                segms = self.model.mask_head(hs, mask_rois, mask_roi_indices)
                mask_loss = 0 * F.sum(segms)
        loss = (rpn_loc_loss + rpn_conf_loss +
                head_loc_loss + head_conf_loss + mask_loss)
        chainer.reporter.report({
            'loss': loss,
            'loss/rpn/loc': rpn_loc_loss, 'loss/rpn/conf': rpn_conf_loss,
            'loss/bbox_head/loc': head_loc_loss,
            'loss/bbox_head/conf': head_conf_loss,
            'loss/mask_head': mask_loss},
            self)
        return loss


def converter(batch, device=None):
    # do not send data to gpu (device is ignored)
    return tuple(list(v) for v in zip(*batch))


def transform_dataset(dataset, model, train):
    if train:
        transform = Compose(
            RGBAugmentation(['rgb']),
            Affine(
                rgb_indices=['rgb'],
                mask_indices=['masks'],
                bbox_indices=['bboxes'],
            ),
            ClassIds2FGClassIds(['labels']),
            AsType(['rgb', 'labels', 'bboxes'],
                   [np.float32, np.int32, np.float32]),
            HWC2CHW(['rgb']),
            Dict2Tuple(['rgb', 'masks', 'labels', 'bboxes']),
            MaskRCNNTransform(800, 1333, model.extractor.mean),
        )
    else:
        transform = Compose(
            ClassIds2FGClassIds(['labels']),
            AsType(['rgb', 'labels', 'bboxes'],
                   [np.float32, np.int32, np.float32]),
            HWC2CHW(['rgb']),
            Dict2Tuple(['rgb', 'masks', 'labels']),
        )
    return TransformDataset(dataset, transform)


def _copyparams(dst, src):
    if isinstance(dst, chainer.Chain):
        for link in dst.children():
            _copyparams(link, src[link.name])
    elif isinstance(dst, chainer.ChainList):
        for i, link in enumerate(dst):
            _copyparams(link, src[i])
    else:
        try:
            dst.copyparams(src)
        except ValueError:
            pass


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--batchsize', type=int, default=16, help='batch size')
    parser.add_argument('--out', default='logs', help='logs')
    parser.add_argument('--resume', help='resume')
    args = parser.parse_args()

    # https://docs.chainer.org/en/stable/chainermn/tutorial/tips_faqs.html#using-multiprocessiterator
    if hasattr(multiprocessing, 'set_start_method'):
        multiprocessing.set_start_method('forkserver')
        p = multiprocessing.Process()
        p.start()
        p.join()

    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank

    class_names = morefusion.datasets.ycb_video.class_names
    fg_class_names = class_names[1:]
    model = MaskRCNNFPNResNet50(
        n_fg_class=len(fg_class_names),
        pretrained_model='imagenet')
    model_coco = MaskRCNNFPNResNet50(pretrained_model='coco')
    _copyparams(model, model_coco)

    model.use_preset('evaluate')
    train_chain = TrainChain(model)
    chainer.cuda.get_device_from_id(device).use()
    train_chain.to_gpu()

    if comm.rank == 0:
        train = chainer.datasets.ConcatenatedDataset(
            morefusion.datasets.YCBVideoInstanceSegmentationDataset(
                split='train', sampling=15
            ),
            morefusion.datasets.YCBVideoSyntheticInstanceSegmentationDataset(
                bg_composite=True
            ),
            morefusion.datasets.MySyntheticYCB20190916InstanceSegmentationDataset(  # NOQA
                'train', bg_composite=True
            ),
        )
        train = transform_dataset(train, model, train=True)
        val = morefusion.datasets.YCBVideoInstanceSegmentationDataset(
            split='keyframe', sampling=1
        )
        val = transform_dataset(val, model, train=False)
    else:
        train = None
        val = None
    train = chainermn.scatter_dataset(train, comm, shuffle=True)
    val = chainermn.scatter_dataset(val, comm, shuffle=False)

    train_iter = chainer.iterators.MultiprocessIterator(
        train, args.batchsize // comm.size,
        n_processes=args.batchsize // comm.size,
        shared_mem=100 * 1000 * 1000 * 4)
    val_iter = chainer.iterators.MultiprocessIterator(
        val, args.batchsize // comm.size,
        n_processes=args.batchsize // comm.size,
        shared_mem=100 * 1000 * 1000 * 4,
        shuffle=False, repeat=False)

    optimizer = chainermn.create_multi_node_optimizer(
        chainer.optimizers.MomentumSGD(), comm)
    optimizer.setup(train_chain)
    optimizer.add_hook(WeightDecay(0.0001))

    for link in model.links():
        if isinstance(link, L.BatchNormalization):
            link.disable_update()
    model.extractor.disable_update()
    model.rpn.disable_update()

    for name, link in model.namedlinks():
        print(name, link.update_enabled)

    updater = training.updaters.StandardUpdater(
        train_iter, optimizer, converter=converter, device=device)
    max_epoch = (180e3 * 8) / 118287
    trainer = training.Trainer(
        updater, (max_epoch, 'epoch'), args.out)

    @make_shift('lr')
    def lr_schedule(trainer):
        base_lr = 0.02 * args.batchsize / 16
        warm_up_duration = 500
        warm_up_rate = 1 / 3

        iteration = trainer.updater.iteration
        if iteration < warm_up_duration:
            rate = warm_up_rate \
                + (1 - warm_up_rate) * iteration / warm_up_duration
        else:
            rate = 1
            for step in [120e3 / 180e3 * max_epoch, 160e3 / 180e3 * max_epoch]:
                if trainer.updater.epoch_detail >= step:
                    rate *= 0.1

        return base_lr * rate

    trainer.extend(lr_schedule)

    val_interval = 10000, 'iteration'
    evaluator = InstanceSegmentationCOCOEvaluator(val_iter, model)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)

    if comm.rank == 0:
        log_interval = 10, 'iteration'
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=log_interval)
        keys = ['epoch', 'iteration', 'lr', 'main/loss',
                'main/loss/rpn/loc', 'main/loss/rpn/conf',
                'main/loss/bbox_head/loc', 'main/loss/bbox_head/conf',
                'main/loss/mask_head',
                'validation/main/map/iou=0.50:0.95/area=all/max_dets=100']
        trainer.extend(extensions.PrintReport(keys), trigger=log_interval)
        trainer.extend(extensions.ProgressBar(update_interval=10))

        # trainer.extend(extensions.snapshot(), trigger=(10000, 'iteration'))
        trainer.extend(
            extensions.snapshot_object(
                model, 'model_iter_best'),
            trigger=training.triggers.MaxValueTrigger(
                'validation/main/map/iou=0.50:0.95/area=all/max_dets=100',
                trigger=val_interval))
        trainer.extend(
            extensions.snapshot_object(
                model, 'model_iter_{.updater.iteration}'),
            trigger=(max_epoch, 'epoch'))

    if args.resume:
        serializers.load_npz(args.resume, trainer, strict=False)

    trainer.run()


if __name__ == '__main__':
    main()
