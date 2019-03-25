#!/usr/bin/env python

import copy

import imgviz
import numpy as np
import pyglet
import trimesh

import objslampp


class MainApp(object):

    def __init__(self, seed=0, split='train', class_ids=(2,), voxel_dim=32):
        np.random.seed(seed)

        self._dataset = \
            objslampp.datasets.YCBVideoMultiViewAlignmentDataset(
                split=split, class_ids=class_ids,
            )
        self._dataset.voxel_dim = voxel_dim

    def _get_data(self, index):
        data = self._dataset[index]
        class_names = objslampp.datasets.ycb_video.class_names
        class_name = class_names[data['class_id']]
        print(f'class_name: {class_name}')
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                if v.size < 5:
                    print(f'{k}: {(v.shape, v)}')
                else:
                    print(f'{k}: {(v.shape, v.dtype)}')
            else:
                print(f'{k}: {v}')
        return data

    def cad(self, data=None, index=200):
        if not data:
            data = self._get_data(index=index)

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

        objslampp.extra.trimesh.show_with_rotation(
            scene=scene, caption='cad point clouds',
        )

    def scan_2d(self, data=None, index=200):
        if not data:
            data = self._get_data(index=index)

        depth2rgb_z = imgviz.Depth2RGB()
        vizs = []
        for rgb, pcd, mask in zip(
            data['scan_rgbs'], data['scan_pcds'], data['scan_masks']
        ):
            mask = mask.astype(np.uint8) * 255
            pcd_z_viz = depth2rgb_z(pcd[:, :, 2])
            viz = imgviz.tile(
                [rgb, pcd_z_viz, mask], (1, 3), border=(255, 255, 255)
            )
            vizs.append(viz)
        vizs = imgviz.tile(vizs, (5, 2), border=(255, 255, 255))
        vizs = imgviz.resize(vizs, height=1000)
        imgviz.io.pyglet_imshow(vizs, 'scan views')
        pyglet.app.run()

    def scan(self, data=None, index=200, color='rgb'):
        if not data:
            data = self._get_data(index=index)

        assert color in ['rgb', 'view'], 'color must be rgb or view'

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

        objslampp.extra.trimesh.show_with_rotation(
            scene, caption='scan ({})'.format(color)
        )

    def cad_voxel_mapping(self, data=None, index=200, show=True):
        if not data:
            data = self._get_data(index=index)

        cad_mapping = objslampp.geometry.VoxelMapping(
            origin=np.array(
                (- self._dataset.voxel_dim // 2 * data['pitch'],) * 3,
                dtype=float
            ),
            pitch=data['pitch'],
            voxel_size=self._dataset.voxel_dim,
            nchannel=3,
        )
        for rgb, pcd in zip(data['cad_rgbs'], data['cad_pcds']):
            isnan = np.isnan(pcd).any(axis=2)
            cad_mapping.add(
                points=pcd[~isnan],
                values=rgb[~isnan].astype(float) / 255,
            )
        if show:
            geom = cad_mapping.as_boxes()
            scene = trimesh.Scene(geom)
            objslampp.extra.trimesh.show_with_rotation(
                scene, caption='cad voxel mapping'
            )
        else:
            return cad_mapping

    def scan_voxel_mapping(self, data=None, index=200, show=True):
        if not data:
            data = self._get_data(index=index)

        scan_mapping = objslampp.geometry.VoxelMapping(
            origin=data['scan_origin'],
            pitch=data['pitch'],
            voxel_size=self._dataset.voxel_dim,
            nchannel=3,
        )
        for rgb, pcd, mask in zip(
            data['scan_rgbs'], data['scan_pcds'], data['scan_masks']
        ):
            isnan = np.isnan(pcd).any(axis=2)
            mask = (~isnan) & mask
            scan_mapping.add(
                points=pcd[mask],
                values=rgb[mask].astype(float) / 255,
            )
        if show:
            geom = scan_mapping.as_boxes()
            scene = trimesh.Scene(geom)
            objslampp.extra.trimesh.show_with_rotation(
                scene, caption='scan voxel mapping'
            )
        else:
            return scan_mapping

    def alignment(self, data=None, index=200):
        if not data:
            data = self._get_data(index=index)

        masks = data['scan_masks']
        rgbs = data['scan_rgbs']
        vizs = []
        for rgb, mask in zip(rgbs, masks):
            bbox = objslampp.geometry.masks_to_bboxes([mask])[0]
            viz = imgviz.instances2rgb(
                rgb, labels=[1], bboxes=[bbox], masks=[mask]
            )
            vizs.append(viz)
        vizs = imgviz.tile(vizs)
        vizs = imgviz.resize(vizs, width=1000)
        imgviz.io.pyglet_imshow(vizs)

        cad_mapping = self.cad_voxel_mapping(data=data, show=False)
        scan_mapping = self.scan_voxel_mapping(data=data, show=False)

        # ---------------------------------------------------------------------
        # initial

        axis_base = trimesh.creation.axis(origin_size=0.02)
        geom_cad = cad_mapping.as_boxes()
        geom_scan = scan_mapping.as_boxes()
        box_cad = cad_mapping.as_bbox(
            face_color=(1.0, 0, 0, 0.5),
            origin_color=(1.0, 0, 0),
        )
        box_scan = scan_mapping.as_bbox(
            face_color=(0, 0, 1.0, 0.5),
            origin_color=(0, 0, 1.0),
        )

        def show(scene, caption):
            return objslampp.extra.trimesh.show_with_rotation(
                scene=scene,
                caption=caption,
                resolution=(500, 500),
                start_loop=False,
            )

        scene = trimesh.Scene([
            axis_base, geom_cad, geom_scan, box_cad, box_scan
        ])
        show(scene, caption='initial')

        # ---------------------------------------------------------------------
        # rotated

        rotation = data['T_cam2cad'].copy()
        rotation[:3, 3] = 0

        geom_scan_rotated = geom_scan.copy()
        geom_scan_rotated.apply_transform(rotation)
        box_scan_rotated = copy.deepcopy(box_scan)
        for geom in box_scan_rotated:
            geom.apply_transform(rotation)

        scene = trimesh.Scene([
            axis_base,
            geom_cad,
            geom_scan_rotated,
            box_cad,
            box_scan_rotated,
        ])
        show(scene=scene, caption='rotated')

        # ---------------------------------------------------------------------
        # transformed

        translation = data['T_cam2cad'][:3, 3]

        geom_scan_transformed = geom_scan_rotated.copy()
        geom_scan_transformed.apply_translation(translation)
        box_scan_transformed = copy.deepcopy(box_scan)
        for geom in box_scan_transformed:
            geom.apply_transform(rotation)
            geom.apply_translation(translation)

        scene = trimesh.Scene([
            axis_base,
            geom_cad,
            geom_scan_transformed,
            box_cad,
            box_scan_transformed,
        ])
        show(scene=scene, caption='transformed')

        # ---------------------------------------------------------------------
        # concatenated

        translation = data['cad_origin'] - data['scan_origin']

        geom_scan_concat = geom_scan.copy()
        geom_scan_concat.apply_translation(translation)
        box_scan_concat = copy.deepcopy(box_scan)
        for geom in box_scan_concat:
            geom.apply_translation(translation)

        scene = trimesh.Scene([
            axis_base,
            geom_cad,
            geom_scan_concat,
            box_cad,
            box_scan_concat,
        ])
        show(scene=scene, caption='concat')

        # ---------------------------------------------------------------------

        rotation = trimesh.transformations.quaternion_matrix(
            data['gt_quaternion']
        )

        geom_scan_concat_rotated = geom_scan.copy()
        geom_scan_concat_rotated.apply_translation(translation)
        geom_scan_concat_rotated.apply_transform(rotation)
        box_scan_concat_rotated = copy.deepcopy(box_scan)
        for geom in box_scan_concat_rotated:
            geom.apply_translation(translation)
            geom.apply_transform(rotation)

        scene = trimesh.Scene([
            axis_base,
            geom_cad,
            geom_scan_concat_rotated,
            box_cad,
            box_scan_concat_rotated,
        ])
        show(scene=scene, caption='concat & rotated')

        # ---------------------------------------------------------------------

        translation_delta = (
            data['gt_translation'] * self._dataset.voxel_dim * data['pitch']
        )

        geom_scan_concat_transformed = geom_scan.copy()
        geom_scan_concat_transformed.apply_translation(translation)
        geom_scan_concat_transformed.apply_translation(translation_delta)
        geom_scan_concat_transformed.apply_transform(rotation)
        box_scan_concat_transformed = copy.deepcopy(box_scan)
        for geom in box_scan_concat_transformed:
            geom.apply_translation(translation)
            geom.apply_translation(translation_delta)
            geom.apply_transform(rotation)

        scene = trimesh.Scene([
            axis_base,
            geom_cad,
            geom_scan_concat_transformed,
            box_cad,
            box_scan_concat_transformed,
        ])
        show(scene=scene, caption='concat & transformed')

        # ---------------------------------------------------------------------

        pyglet.app.run()


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
