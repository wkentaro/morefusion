#!/usr/bin/env python

import imgviz
import numpy as np
import termcolor

import morefusion


class MainApp(object):
    def __init__(self):
        self._dataset = morefusion.datasets.YCBVideoDataset(split="train")
        self._mainloop()

    def _mainloop(self):
        class Images(list):
            def __init__(self, app):
                self.app = app
                self.index = 1000

            def __len__(self):
                return len(self.app._dataset)

            def __getitem__(self, index):
                image_id = self.app._dataset.ids[index]
                termcolor.cprint(f"[{index}] {image_id}", attrs={"bold": True})

                frame = self.app._dataset[index]
                image = self.app._process_frame(frame)
                return image

        imgviz.io.pyglet_imshow(Images(app=self))
        imgviz.io.pyglet_run()

    def _process_frame(self, frame):
        meta = frame["meta"]
        color = frame["color"]

        depth = frame["depth"]
        depth_viz = imgviz.depth2rgb(depth, min_value=0, max_value=2)

        label = frame["label"]
        label_viz = imgviz.label2rgb(label)

        labels = meta["cls_indexes"].astype(np.int32)
        # NOTE: cls_mask is the same as ins_mask in YCB_Video_Dataset
        masks = np.asarray([label == cls_id for cls_id in labels])
        bboxes = morefusion.geometry.masks_to_bboxes(masks)

        keep = ~(bboxes == 0).all(axis=1)
        labels = labels[keep]
        bboxes = bboxes[keep]
        masks = masks[keep]

        gray = imgviz.gray2rgb(imgviz.rgb2gray(color))
        ins_viz = imgviz.instances2rgb(
            gray, labels=labels, bboxes=bboxes, masks=masks
        )

        vertmap = meta["vertmap"]
        vertmap[label == 0] = np.nan
        vert_viz_x = imgviz.depth2rgb(vertmap[:, :, 0])
        vert_viz_y = imgviz.depth2rgb(vertmap[:, :, 1])
        vert_viz_z = imgviz.depth2rgb(vertmap[:, :, 2])

        roi_viz_color = []
        roi_viz_depth = []
        roi_viz_label = []
        for bbox, mask in zip(bboxes, masks):
            y1, x1, y2, x2 = bbox.round().astype(int)
            mask_roi = mask[y1:y2, x1:x2]
            color_roi = color[y1:y2, x1:x2].copy()
            color_roi[~mask_roi] = 0
            depth_roi = depth_viz[y1:y2, x1:x2].copy()
            depth_roi[~mask_roi] = 0
            label_roi = label_viz[y1:y2, x1:x2].copy()
            label_roi[~mask_roi] = 0
            roi_viz_color.append(color_roi)
            roi_viz_depth.append(depth_roi)
            roi_viz_label.append(label_roi)
        roi_viz_color = imgviz.tile(roi_viz_color, border=(255, 255, 255))
        roi_viz_depth = imgviz.tile(roi_viz_depth, border=(255, 255, 255))
        roi_viz_label = imgviz.tile(roi_viz_label, border=(255, 255, 255))

        viz = imgviz.tile(
            [
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
                np.zeros_like(roi_viz_color),
            ],
            shape=(3, 4),
            border=(255, 255, 255),
        )
        viz = imgviz.centerize(viz, (1000, 1000))

        return viz


if __name__ == "__main__":
    MainApp()
