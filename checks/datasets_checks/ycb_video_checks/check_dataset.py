#!/usr/bin/env python

import imgviz
import numpy as np
import termcolor

import objslampp


class MainApp(object):

    def __init__(self):
        self._dataset = objslampp.datasets.YCBVideoDataset(split='train')
        self._mainloop()

    def _mainloop(self):
        index = 1000

        play = False
        while True:
            image_id = self._dataset.ids[index]
            termcolor.cprint(f'[{index}] {image_id}', attrs={'bold': True})

            frame = self._dataset[index]
            image = self._process_frame(frame)

            imgviz.io.cv_imshow(image, __file__)

            if play:
                key = imgviz.io.cv_waitkey(1)
                index += 15
            else:
                key = imgviz.io.cv_waitkey()

            if key >= 0:
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

    def _process_frame(self, frame):
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

        feat_viz = np.zeros_like(color)

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
    MainApp()
