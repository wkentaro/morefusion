#!/usr/bin/env python

import json
import path

import chainer
from chainer import cuda
import easydict
import imgviz
import numpy as np

from morefusion.contrib import singleview_3d


def get_images():
    log_dir = path.Path('logs.20190822/20190825_015100.635309989')

    with open(log_dir / 'args') as f:
        args = easydict.EasyDict(json.load(f))

    model = singleview_3d.models.BaselineModel(
        n_fg_class=len(args.class_names[1:]),
    )
    model.to_gpu(0)

    chainer.serializers.load_npz(
        log_dir / 'snapshot_model_best_auc_add.npz', model
    )

    dataset = singleview_3d.datasets.YCBVideoDataset(
        'train', class_ids=args.class_ids
    )

    nchannel2rgb = imgviz.Nchannel2RGB()
    for index in range(100, 200):
        examples = dataset.get_example(index)

        chainer.config.train = False

        in_data = chainer.dataset.concat_examples(examples, device=0)

        print(f"index: {index}, class_id: {in_data['class_id']}")

        values, points = model._extract(
            in_data['rgb'],
            in_data['pcd'],
        )

        feats = []
        for i in range(len(values)):
            mask = ~np.isnan(cuda.to_cpu(in_data['pcd'][i])).any(axis=2)

            values_i = cuda.to_cpu(values[i].array)
            _, C = values_i.shape
            H, W = in_data['rgb'].shape[1:3]
            feat = np.zeros((H, W, C), dtype=values_i.dtype)
            feat[mask] = values_i
            feats.append(feat)

        if nchannel2rgb.pca is None:
            nchannel2rgb(np.hstack(feats))

        vizs = []
        for i in range(len(values)):
            rgb = cuda.to_cpu(in_data['rgb'][i])
            feat = feats[i]
            feat_viz = nchannel2rgb(feat, dtype=np.uint8)
            viz = imgviz.tile([rgb, feat_viz])
            vizs.append(viz)
        yield imgviz.tile(vizs, border=(255, 255, 255))


def main():
    imgviz.io.pyglet_imshow(get_images())
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
