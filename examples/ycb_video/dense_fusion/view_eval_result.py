#!/usr/bin/env python

import argparse

import chainer
import gdown
import imgviz
import numpy as np
import path
import pybullet
import scipy.io
import trimesh

import objslampp


def get_eval_result(refine):
    root_dir = chainer.dataset.get_dataset_directory('wkentaro/objslampp/ycb_video/dense_fusion/eval_result/ycb')  # NOQA
    root_dir = path.Path(root_dir)

    if refine:
        zip_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=1sqgpOgcFJB0P4Nx5vwHN8vKj_R0Ng0S0',  # NOQA
            path=root_dir / 'Densefusion_iterative_result.zip',
            postprocess=gdown.extractall,
            md5='46a8b4a16de5b2f87c969c793b1d1825',
        )
    else:
        zip_file = gdown.cached_download(
            url='https://drive.google.com/uc?id=179MtZOvKNR1aJ310GIv0Gi2yf9nb13q3',  # NOQA
            path=root_dir / 'Densefusion_wo_refine_result.zip',
            postprocess=gdown.extractall,
            md5='fbe2524635c44e64af94ab7cf7a19e9d',
        )
    result_dir = path.Path(zip_file[:- len('.zip')])

    return result_dir


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--refine', action='store_true')
    args = parser.parse_args()

    result_dir = get_eval_result(args.refine)

    class Images:

        offset = 0
        step = 15
        indices = np.arange(offset, 2948, step)

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, index):
            index = self.indices[index]
            result_file = result_dir / f'{index:04d}.mat'  # NOQA
            print(result_file)
            result = scipy.io.loadmat(
                result_file, chars_as_strings=True, squeeze_me=True
            )
            frame_id = '/'.join(result['frame_id'].split('/')[1:])

            frame = objslampp.datasets.YCBVideoDataset.get_frame(frame_id)

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

            captions = [objslampp.datasets.ycb_video.class_names[l]
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
                cad_file = objslampp.datasets.YCBVideoModels().get_cad_model(
                    class_id=class_id
                )
                objslampp.extra.pybullet.add_model(
                    cad_file,
                    position=pose[4:],
                    orientation=pose[:4][[1, 2, 3, 0]],
                )
            rgb_rend, depth_rend, segm_rend = \
                objslampp.extra.pybullet.render_camera(
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
