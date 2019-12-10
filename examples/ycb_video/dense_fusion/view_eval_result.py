#!/usr/bin/env python

import argparse

import imgviz
import numpy as np
import path
import pybullet
import scipy.io
import trimesh

import morefusion


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--result',
        default='/home/wkentaro/data/datasets/wkentaro/morefusion/ycb_video/dense_fusion/eval_result/ycb/Densefusion_wo_refine_result',  # NOQA
        help='result dir',
    )
    parser.add_argument(
        '--step',
        type=int,
        default=1,
        help='step',
    )
    args = parser.parse_args()

    args.result = path.Path(args.result)

    class Images:

        offset = 0
        step = args.step
        result_files = args.result.glob('*.mat')
        indices = np.arange(offset, len(result_files), step)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            index = self.indices[index]
            result_file = args.result / f'{index:04d}.mat'  # NOQA
            print(result_file)
            result = scipy.io.loadmat(
                result_file, chars_as_strings=True, squeeze_me=True
            )
            frame_id = '/'.join(result['frame_id'].split('/')[1:])

            frame = morefusion.datasets.YCBVideoDataset.get_frame(frame_id)

            rgb = frame['color']
            depth = frame['depth']
            bboxes = result['bboxes']
            K = frame['meta']['intrinsic_matrix']
            labels = result['labels'].astype(np.int32)
            masks = result['masks'].astype(bool)

            keep = np.isin(labels, frame['meta']['cls_indexes'])
            bboxes = bboxes[keep]
            labels = labels[keep]
            masks = masks[keep]

            captions = [morefusion.datasets.ycb_video.class_names[l]
                        for l in labels]
            detections_viz = imgviz.instances2rgb(
                rgb,
                labels=labels,
                bboxes=bboxes,
                masks=masks,
                captions=captions,
                font_size=15,
            )

            camera = trimesh.scene.Camera(
                resolution=(640, 480), focal=(K[0, 0], K[1, 1]))

            pybullet.connect(pybullet.DIRECT)
            for class_id, pose in zip(labels, result['poses']):
                cad_file = morefusion.datasets.YCBVideoModels().get_cad_file(
                    class_id=class_id
                )
                morefusion.extra.pybullet.add_model(
                    cad_file,
                    position=pose[4:],
                    orientation=pose[:4][[1, 2, 3, 0]],
                )
            rgb_rend, depth_rend, segm_rend = \
                morefusion.extra.pybullet.render_camera(
                    np.eye(4), fovy=camera.fov[1], height=480, width=640
                )
            pybullet.disconnect()

            min_value = 0.3
            max_value = 2 * np.nanmedian(depth) - min_value
            depth = imgviz.depth2rgb(
                depth, min_value=min_value, max_value=max_value)
            depth_rend = imgviz.depth2rgb(
                depth_rend, min_value=min_value, max_value=max_value)

            viz = imgviz.tile(
                [rgb, depth, detections_viz, rgb_rend, depth_rend],
                (2, 3),
                border=(255,) * 3,
            )
            viz = imgviz.resize(viz, width=1500)
            return viz

    imgviz.io.pyglet_imshow(Images())
    imgviz.io.pyglet_run()


if __name__ == '__main__':
    main()
