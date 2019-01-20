#!/usr/bin/env python

import argparse

from chainer.backends import cuda
from chainercv.links.model.resnet import ResNet50
import imgviz
import numpy as np
import termcolor

import objslampp


class MainApp(object):

    def __init__(self):
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument('--gpu', type=int, default=0, help='gpu id')
        args = parser.parse_args()

        self.dataset = objslampp.datasets.YCBVideoDataset()

        if args.gpu >= 0:
            cuda.get_device_from_id(args.gpu).use()
        self.resnet = ResNet50(pretrained_model='imagenet', arch='he')
        self.resnet.pick = ['res4']
        if args.gpu >= 0:
            self.resnet.to_gpu()
        self.nchannel2rgb = imgviz.Nchannel2RGB()

    def extract_feature(self, rgb):
        x = rgb.transpose(2, 0, 1)
        x = x - self.resnet.mean
        x = x[None]
        if self.resnet.xp != np:
            x = cuda.to_gpu(x)
        feat, = self.resnet(x)
        feat = cuda.to_cpu(feat[0].array)
        return feat.transpose(1, 2, 0)

    def feature2rgb(self, feat, mask_fg):
        dst = self.nchannel2rgb(feat, dtype=float)
        H, W = mask_fg.shape[:2]
        dst = imgviz.resize(dst, height=H, width=W)
        dst = (dst * 255).astype(np.uint8)
        dst[~mask_fg] = 0
        return dst

    def run(self):
        index = 0
        imageset = self.dataset.imageset('train')

        play = False
        while True:
            image_id = imageset[index]
            termcolor.cprint(f'[{index}] {image_id}', attrs={'bold': True})

            frame = self.dataset.get_frame(image_id)
            image = self.process_frame(frame)

            imgviz.io.cv_imshow(image, __file__)

            if play:
                key = imgviz.io.cv_waitkey(1)
                index += 15
            else:
                key = imgviz.io.cv_waitkey()

            try:
                key = chr(key)
            except Exception:
                pass

            if key in list('qs'):
                print(f'key: {key}')

            if key == 's':
                play = not play
            elif key == 'q':
                break

    def process_frame(self, frame):
        meta = frame['meta']
        color = frame['color']

        depth = frame['depth']
        depth_viz = imgviz.depth2rgb(depth, min_value=0, max_value=2)

        label = frame['label']
        label_viz = imgviz.label2rgb(label)

        labels = meta['cls_indexes']
        # NOTE: cls_mask is the same as ins_mask in YCB_Video_Dataset
        masks = [label == cls_id for cls_id in labels]
        bboxes = imgviz.instances.mask_to_bbox(masks)
        gray = imgviz.gray2rgb(imgviz.rgb2gray(color))
        ins_viz = imgviz.instances2rgb(
            gray, labels=labels, bboxes=bboxes, masks=masks
        )

        vertmap = meta['vertmap']
        vertmap[label == 0] = np.nan
        vert_viz_x = imgviz.depth2rgb(vertmap[:, :, 0])
        vert_viz_y = imgviz.depth2rgb(vertmap[:, :, 1])
        vert_viz_z = imgviz.depth2rgb(vertmap[:, :, 2])

        feat = self.extract_feature(color)
        feat_viz = self.feature2rgb(feat, label != 0)

        roi_viz_color = []
        roi_viz_depth = []
        roi_viz_label = []
        roi_viz_feat = []
        for bbox, mask in zip(bboxes, masks):
            y1, x1, y2, x2 = bbox.round().astype(int)
            mask_roi = mask[y1:y2, x1:x2]
            color_roi = color[y1:y2, x1:x2].copy()
            color_roi[~mask_roi] = 0
            depth_roi = depth_viz[y1:y2, x1:x2].copy()
            depth_roi[~mask_roi] = 0
            label_roi = label_viz[y1:y2, x1:x2].copy()
            label_roi[~mask_roi] = 0
            feat_roi = feat_viz[y1:y2, x1:x2].copy()
            feat_roi[~mask_roi] = 0
            roi_viz_color.append(color_roi)
            roi_viz_depth.append(depth_roi)
            roi_viz_label.append(label_roi)
            roi_viz_feat.append(feat_roi)
        roi_viz_color = imgviz.tile(roi_viz_color, border=(255, 255, 255))
        roi_viz_depth = imgviz.tile(roi_viz_depth, border=(255, 255, 255))
        roi_viz_label = imgviz.tile(roi_viz_label, border=(255, 255, 255))
        roi_viz_feat = imgviz.tile(roi_viz_feat, border=(255, 255, 255))

        viz = imgviz.tile([
            color,
            depth_viz,
            label_viz,
            ins_viz,
            vert_viz_x,
            vert_viz_y,
            vert_viz_z,
            np.zeros_like(color),
            roi_viz_color,
            roi_viz_depth,
            roi_viz_label,
            roi_viz_feat,
        ], shape=(3, 4), border=(255, 255, 255))
        viz = imgviz.resize(viz, height=500)

        return viz


if __name__ == '__main__':
    MainApp().run()
