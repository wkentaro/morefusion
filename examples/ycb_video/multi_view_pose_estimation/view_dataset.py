#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh
import trimesh.viewer

import objslampp


class MainApp(object):

    def __init__(self, seed=0):
        np.random.seed(seed)

        self._dataset = \
            objslampp.datasets.YCBVideoMultiViewPoseEstimationDataset(
                split='train', class_ids=[15],
            )

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

    def cad(self):
        data = self._get_data(index=200)

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
        data = self._get_data(index=200)

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

        data = self._get_data(index=200)

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

    def cad_voxel_mapping(self, data=None, show=True):
        if data is None:
            data = self._get_data(index=200)

        cad_mapping = objslampp.geometry.VoxelMapping(
            origin=np.array((-16 * data['pitch'],) * 3, dtype=float),
            pitch=data['pitch'],
            voxel_size=32,
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
            geom.show()
        else:
            return cad_mapping

    def scan_voxel_mapping(self, data=None, show=True):
        if data is None:
            data = self._get_data(index=200)

        scan_mapping = objslampp.geometry.VoxelMapping(
            origin=data['scan_origin'],
            pitch=data['pitch'],
            voxel_size=32,
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
            geom.show()
        else:
            return scan_mapping

    def alignment(self, data=None):
        if data is None:
            data = self._get_data(index=200)

        axis_base = trimesh.creation.axis(origin_size=0.02)

        scan_rgbs = imgviz.tile(
            list(data['scan_rgbs']), border=(255, 255, 255)
        )
        scan_rgbs = imgviz.resize(scan_rgbs, width=1000)
        imgviz.io.pyglet_imshow(scan_rgbs)

        cad_mapping = self.cad_voxel_mapping(data=data, show=False)
        scan_mapping = self.scan_voxel_mapping(data=data, show=False)

        geom_cad = cad_mapping.as_boxes()
        geom_scan = scan_mapping.as_boxes()

        # box_center = (0, 0, 0)
        box_cad = trimesh.creation.box(cad_mapping.voxel_bbox_extents)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['cad_origin'])
        box_cad += voxel_origin
        box_cad.visual.face_colors = (1.0, 0, 0, 0.5)

        box_center = data['scan_origin'] + data['pitch'] * 16
        box_scan = trimesh.creation.box(scan_mapping.voxel_bbox_extents)
        box_scan.apply_translation(box_center)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['scan_origin'])
        box_scan += voxel_origin
        box_scan.visual.face_colors = (0, 0, 1.0, 0.5)

        def show(scene, **kwargs):

            resolution = kwargs.pop('resolution', (500, 500))

            def callback(scene):
                if hasattr(scene, 'angles'):
                    scene.angles += [0, np.deg2rad(1), 0]
                else:
                    scene.angles = np.zeros(3)
                scene.set_camera(angles=scene.angles)

            return trimesh.viewer.SceneViewer(
                scene=scene, callback=callback, resolution=resolution, **kwargs
            )

        scene = trimesh.Scene([
            axis_base, geom_cad, geom_scan, box_cad, box_scan
        ])
        show(scene, caption='initial', start_loop=False)

        # ---------------------------------------------------------------------

        geom_cad_rotated = geom_cad.copy()
        T = data['gt_pose'].copy()
        T[:3, 3] = 0
        geom_cad_rotated.apply_transform(T)

        box_cad = trimesh.creation.box(cad_mapping.voxel_bbox_extents)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['cad_origin'])
        box_cad += voxel_origin
        box_cad.apply_transform(T)
        box_cad.visual.face_colors = (1.0, 0, 0, 0.5)

        scene = trimesh.Scene([
            axis_base, geom_cad_rotated, geom_scan, box_cad, box_scan
        ])
        show(scene=scene, caption='rotated', start_loop=False)

        # ---------------------------------------------------------------------

        geom_cad_transformed = geom_cad_rotated.copy()
        geom_cad_transformed.apply_translation(data['gt_pose'][:3, 3])

        box_cad = trimesh.creation.box(cad_mapping.voxel_bbox_extents)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['cad_origin'])
        box_cad += voxel_origin
        box_cad.apply_transform(T)
        box_cad.apply_translation(data['gt_pose'][:3, 3])
        box_cad.visual.face_colors = (1.0, 0, 0, 0.5)

        scene = trimesh.Scene([
            axis_base, geom_cad_transformed, geom_scan, box_cad, box_scan
        ])
        show(scene=scene, caption='transformed', start_loop=False)

        # ---------------------------------------------------------------------

        geom_cad_concat = geom_cad.copy()
        geom_cad_concat.apply_translation(
            data['scan_origin'] - data['cad_origin']
        )

        box_cad = trimesh.creation.box(cad_mapping.voxel_bbox_extents)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['cad_origin'])
        box_cad += voxel_origin
        box_cad.apply_translation(data['scan_origin'] - data['cad_origin'])
        box_cad.visual.face_colors = (1.0, 0, 0, 0.5)

        scene = trimesh.Scene([
            axis_base, geom_cad_concat, geom_scan, box_cad, box_scan
        ])
        show(scene=scene, caption='concat', start_loop=False)

        # ---------------------------------------------------------------------

        T = trimesh.transformations.quaternion_matrix(data['gt_quaternion'])

        geom_cad_concat_rotated = geom_cad.copy()
        geom_cad_concat_rotated.apply_transform(T)
        geom_cad_concat_rotated.apply_translation(
            data['scan_origin'] - data['cad_origin']
        )

        box_cad = trimesh.creation.box(cad_mapping.voxel_bbox_extents)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['cad_origin'])
        box_cad += voxel_origin
        box_cad.apply_transform(T)
        box_cad.apply_translation(data['scan_origin'] - data['cad_origin'])
        box_cad.visual.face_colors = (1.0, 0, 0, 0.5)

        scene = trimesh.Scene([
            axis_base, geom_cad_concat_rotated, geom_scan, box_cad, box_scan
        ])
        show(scene=scene, caption='concat & rotated', start_loop=False)

        # ---------------------------------------------------------------------

        translation = data['gt_translation'] * 32 * data['pitch']

        geom_cad_concat_transformed = geom_cad_concat_rotated.copy()
        geom_cad_concat_transformed.apply_translation(translation)

        box_cad = trimesh.creation.box(cad_mapping.voxel_bbox_extents)
        voxel_origin = trimesh.creation.icosphere(radius=0.01)
        voxel_origin.apply_translation(data['cad_origin'])
        box_cad += voxel_origin
        box_cad.apply_transform(T)
        box_cad.apply_translation(data['scan_origin'] - data['cad_origin'])
        box_cad.apply_translation(translation)
        box_cad.visual.face_colors = (1.0, 0, 0, 0.5)

        scene = trimesh.Scene([
            axis_base,
            geom_cad_concat_transformed,
            geom_scan,
            box_cad,
            box_scan,
        ])
        show(scene=scene, caption='concat & transformed', start_loop=False)

        # ---------------------------------------------------------------------

        import pyglet
        pyglet.app.run()


if __name__ == '__main__':
    import fire

    fire.Fire(MainApp)
