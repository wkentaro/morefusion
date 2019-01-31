#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh

import objslampp


class MainApp(object):

    def __init__(self, seed=0):
        np.random.seed(seed)

        self._dataset = \
            objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
                split='train'
            )

    def _get_data(self, index):
        data = self._dataset[index]
        class_names = objslampp.datasets.ycb_video.class_names
        class_name = class_names[data['class_id']]
        print(f'class_name: {class_name}')
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(f'{k}: {(v.shape, v.dtype)}')
            else:
                print(f'{k}: {v}')
        return data

    def cad(self):
        data = self._get_data(index=16)

        scene = trimesh.Scene()

        # cad origin
        geom = trimesh.creation.axis(
            origin_size=0.01, origin_color=(1.0, 0, 0)
        )
        geom.apply_translation(data['cad_origin'])
        scene.add_geometry(geom)

        # cad point cloud
        for rgb, pcd in zip(data['cad_rgbs'], data['cad_pcds']):
            isnan = np.isnan(pcd).any(axis=2)
            geom = trimesh.PointCloud(vertices=pcd[~isnan], color=rgb[~isnan])
            scene.add_geometry(geom)

        scene.show()

    def scan_2d(self):
        data = self._get_data(index=16)

        depth2rgb_z = imgviz.Depth2RGB()
        for rgb, pcd, mask in zip(
            data['scan_rgbs'], data['scan_pcds'], data['scan_masks']
        ):
            mask = mask.astype(np.uint8) * 255
            pcd_z_viz = depth2rgb_z(pcd[:, :, 2])
            viz = imgviz.tile([rgb, pcd_z_viz, mask])
            viz = imgviz.resize(viz, width=1000)
            imgviz.io.cv_imshow(viz, __file__)
            while True:
                key = imgviz.io.cv_waitkey()
                if key == ord('n'):
                    break
                elif key == ord('q'):
                    return

    def scan(self, color='rgb'):
        assert color in ['rgb', 'view'], 'color must be rgb or view'

        data = self._get_data(index=16)

        scene = trimesh.Scene()

        # world origin
        geom = trimesh.creation.axis(
            origin_size=0.01, origin_color=(1.0, 0, 0)
        )
        scene.add_geometry(geom)

        # scan origin
        geom = trimesh.creation.axis(
            origin_size=0.01, origin_color=(0, 0, 1.0)
        )
        geom.apply_translation(data['scan_origin'])
        scene.add_geometry(geom)

        # scan point cloud
        n_view = len(data['scan_rgbs'])
        for i_view in range(n_view):
            rgb = data['scan_rgbs'][i_view]
            pcd = data['scan_pcds'][i_view]

            isnan = np.isnan(pcd).any(axis=2)
            if color == 'rgb':
                colors = rgb[~isnan]
            else:
                assert color == 'view'
                colormap = imgviz.label_colormap()[1:]  # drop black
                colors = np.full(
                    ((~isnan).sum(), 3),
                    colormap[i_view % len(colormap)]
                )

            geom = trimesh.PointCloud(vertices=pcd[~isnan], color=colors)
            scene.add_geometry(geom)

        # gt pose
        geom = trimesh.creation.axis(
            origin_size=0.01, origin_color=(0, 1.0, 0)
        )
        geom.apply_transform(data['gt_pose'])
        scene.add_geometry(geom)

        scene.show()


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
