#!/usr/bin/env python

import glooey
import numpy as np
import octomap
import imgviz
import trimesh
import trimesh.transformations as tf
import pyglet
import sklearn.neighbors

import objslampp


def leaves_from_tree(tree):
    root = tree.begin_tree()

    occupied = []
    empty = []
    for node in root:
        if node.isLeaf():
            coord = node.getCoordinate()
            if tree.isNodeOccupied(node):
                occupied.append(coord)
            else:
                empty.append(coord)

    occupied = np.array(occupied, dtype=float)
    empty = np.array(empty, dtype=float)
    return occupied, empty


def visualize_occupied_empty_points(
    octrees,
    height,
    width,
    K,
):
    # visualize occupied/empty points
    scene_occupied = trimesh.Scene()
    scene_empty = trimesh.Scene()
    colormap = imgviz.label_colormap()
    for instance_id, octree in octrees.items():
        occupied, empty = leaves_from_tree(octree)
        geom = trimesh.PointCloud(
            vertices=occupied, colors=colormap[instance_id]
        )
        scene_occupied.add_geometry(geom)
        geom = trimesh.PointCloud(vertices=empty, colors=[0.5, 0.5, 0.5])
        scene_empty.add_geometry(geom)
    for scene in [scene_occupied, scene_empty]:
        scene.camera.resolution = (width, height)
        scene.camera.focal = (K[0, 0], K[1, 1])
        scene.camera.transform = objslampp.extra.trimesh.camera_transform(
            tf.translation_matrix([0, 0, -1.5])
        )
    window = pyglet.window.Window(width=640 * 2, height=480)

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()

    def callback(dt):
        for scene in [scene_occupied, scene_empty]:
            scene.camera.transform = tf.rotation_matrix(
                np.deg2rad(1), [0, 1, 0], point=scene_occupied.centroid
            ) @ scene.camera.transform

    gui = glooey.Gui(window)
    hbox = glooey.HBox()
    hbox.set_padding(5)
    vbox = glooey.VBox()
    vbox.add(glooey.Label('occupied', color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene_occupied))
    hbox.add(vbox)
    vbox = glooey.VBox()
    vbox.add(glooey.Label('empty', color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene_empty))
    hbox.add(vbox)
    gui.add(hbox)
    pyglet.clock.schedule_interval(callback, 1 / 30)
    pyglet.app.run()


def visualize_instance_grids(
    instance_ids,
    instance_label,
    instance_extents,
    octrees,
    rgb,
    pcd,
    pitch,
    K,
):
    nonnan = ~np.isnan(pcd).any(axis=2)
    geom = trimesh.PointCloud(vertices=pcd[nonnan], colors=rgb[nonnan])
    scene_pcd = geom.scene()

    scene_occupied = trimesh.Scene()
    scene_empty = trimesh.Scene()
    for instance_id in instance_ids:
        mask = instance_label == instance_id
        extents = instance_extents[instance_id == instance_ids][0]
        grid, aabb_min, aabb_max = get_instance_grid(
            octrees, pitch, pcd, mask, instance_id, extents
        )

        for i, scene in enumerate([scene_occupied, scene_empty]):
            if i == 0:
                # visualize occupied spaces
                voxel = trimesh.voxel.Voxel(
                    ~np.isin(grid, [254, 255]), pitch, aabb_min)
            else:
                voxel = trimesh.voxel.Voxel(
                    np.isin(grid, [254]), pitch, aabb_min)
            colors = imgviz.label2rgb(
                grid.reshape(1, -1)).reshape(grid.shape + (3,))
            alpha = np.full(grid.shape + (1,), 127, dtype=np.uint8)
            colors = np.concatenate((colors, alpha), axis=3)
            geom = voxel.as_boxes(colors=colors)
            scene.add_geometry(geom)
            geom = trimesh.path.creation.box_outline(aabb_max - aabb_min)
            geom.apply_translation((aabb_min + aabb_max) / 2)
            scene.add_geometry(geom)

    for scene in [scene_pcd, scene_occupied, scene_empty]:
        scene.camera.resolution = (640, 480)
        scene.camera.focal = (K[0, 0], K[1, 1])
        scene.camera.transform = objslampp.extra.trimesh.camera_transform(
            tf.translation_matrix([0, 0, -0.5])
        )

    window = pyglet.window.Window(width=640 * 3, height=480)

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()

    def callback(dt):
        for scene in [scene_pcd, scene_occupied, scene_empty]:
            scene.camera.transform = tf.rotation_matrix(
                np.deg2rad(1), [0, 1, 0], point=scene_occupied.centroid
            ) @ scene.camera.transform

    gui = glooey.Gui(window)
    hbox = glooey.HBox()
    hbox.set_padding(5)
    vbox = glooey.VBox()
    vbox.add(glooey.Label('pcd', color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene_pcd))
    hbox.add(vbox)
    vbox = glooey.VBox()
    vbox.add(glooey.Label('occupied', color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene_occupied))
    hbox.add(vbox)
    vbox = glooey.VBox()
    vbox.add(glooey.Label('empty', color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene_empty))
    hbox.add(vbox)
    gui.add(hbox)
    pyglet.clock.schedule_interval(callback, 1 / 30)
    pyglet.app.run()


def get_instance_grid(
    octrees, pitch, pcd, mask, instance_id, extents, threshold=None
):
    nonnan = ~np.isnan(pcd).any(axis=2)
    pcd_ins = pcd[mask & nonnan]
    centroid = pcd_ins.mean(axis=0)
    aabb_min = centroid - extents / 2
    aabb_max = aabb_min + extents
    grid_shape = np.ceil(extents / pitch).astype(int)
    # unknown: 255
    # free: 254
    # occupied background: 0
    # occupied instance1: 1
    # occupied instance2: 2
    # ...
    grid = np.full(grid_shape, 255, dtype=np.uint8)
    centers = trimesh.voxel.matrix_to_points(
        grid == 255, pitch=pitch, origin=aabb_min
    )
    if threshold is None:
        threshold = 2 * np.sqrt(3)
    threshold *= pitch  # thershold in dimension to metric
    for instance_id, octree in octrees.items():
        occupied, empty = leaves_from_tree(octree)
        # empty
        kdtree = sklearn.neighbors.KDTree(empty)
        dist, indices = kdtree.query(centers, k=1)
        grid[dist[:, 0].reshape(grid.shape) < threshold] = 254
        # occupied
        kdtree = sklearn.neighbors.KDTree(occupied)
        dist, indices = kdtree.query(centers, k=1)
        grid[dist[:, 0].reshape(grid.shape) < threshold] = instance_id
    return grid, aabb_min, aabb_max


def main():
    dataset = objslampp.datasets.YCBVideoDataset('train')
    frame = dataset[0]

    class_ids = frame['meta']['cls_indexes']
    instance_ids = class_ids
    K = frame['meta']['intrinsic_matrix']
    rgb = frame['color']
    depth = frame['depth']
    height, width = rgb.shape[:2]
    nonnan = ~np.isnan(depth)
    pcd = objslampp.geometry.pointcloud_from_depth(
        depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
    )
    instance_label = frame['label']

    # build octrees
    pitch = 0.005
    octrees = {}
    for instance_id in np.r_[0, instance_ids]:
        mask = instance_label == instance_id

        octree = octomap.OcTree(pitch)
        octree.insertPointCloud(
            pcd[mask & nonnan],
            np.array([0, 0, 0], dtype=float),
        )
        octrees[instance_id] = octree

    # visualize_occupied_empty_points(
    #     octrees=octrees,
    #     height=height,
    #     width=width,
    #     K=K,
    # )

    models = objslampp.datasets.YCBVideoModels()
    instance_extents = []
    for instance_id, class_id in zip(instance_ids, class_ids):
        cad_file = models.get_cad_model(class_id=class_id)
        diagonal = models.get_bbox_diagonal(cad_file)
        extents = (diagonal,) * 3
        instance_extents.append(extents)
    instance_extents = np.array(instance_extents)

    visualize_instance_grids(
        instance_ids=instance_ids,
        instance_label=instance_label,
        instance_extents=instance_extents,
        octrees=octrees,
        rgb=rgb,
        pcd=pcd,
        pitch=pitch,
        K=K,
    )


if __name__ == '__main__':
    main()
