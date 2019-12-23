#!/usr/bin/env python

import numpy as np
import trimesh
import trimesh.transformations as tf
import trimesh.viewer

import morefusion


def get_scenes():
    dataset = morefusion.datasets.YCBVideoDataset('train')

    index = 0
    video_id_prev = None
    scene = trimesh.Scene()
    while index < len(dataset):
        example = dataset[index]

        video_id, frame_id = dataset.ids[index].split('/')

        clear = False
        if video_id_prev is not None and video_id_prev != video_id:
            clear = True
            for node_name in scene.graph.nodes_geometry:
                scene.graph.transforms.remove_node(node_name)
        video_id_prev = video_id

        rgb = example['color']
        depth = example['depth']
        K = example['meta']['intrinsic_matrix']

        T_world2cam = np.r_[
            example['meta']['rotation_translation_matrix'], [[0, 0, 0, 1]]
        ]
        T_cam2world = np.linalg.inv(T_world2cam)
        pcd = morefusion.geometry.pointcloud_from_depth(
            depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
        )
        nonnan = ~np.isnan(depth)
        geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
        scene.add_geometry(geom, transform=T_cam2world)

        # A kind of current camera view, but a bit far away to see whole scene.
        scene.camera.resolution = (rgb.shape[1], rgb.shape[0])
        scene.camera.focal = (K[0, 0], K[1, 1])
        scene.camera_transform = morefusion.extra.trimesh.to_opengl_transform(
            T_cam2world @ tf.translation_matrix([0, 0, -0.5])
        )
        # scene.set_camera()

        geom = trimesh.creation.camera_marker(
            scene.camera, marker_height=0.05
        )[1]
        geom.colors = (0, 1., 0)
        scene.add_geometry(geom, transform=T_cam2world)

        index += 15
        print(f'[{index:08d}] video_id={video_id}')
        yield {'__clear__': clear, 'rgb': rgb, 'scene': scene}


def main():
    morefusion.extra.trimesh.display_scenes(get_scenes(), tile=(1, 2))


if __name__ == '__main__':
    main()
