#!/usr/bin/env python

import pathlib

import glooey
import numpy as np
import octomap
import imgviz
import trimesh
import sklearn.neighbors
import pyglet

import objslampp


def walktree(tree):
    import queue

    root = tree.begin_tree()

    deque = queue.deque()
    deque.append(root)
    while deque:
        for child in deque.pop():
            if child.isLeaf():
                yield (
                    child.getSize(),
                    child.getCoordinate(),
                    octree.isNodeOccupied(child),
                )
            else:
                deque.append(child)


models = objslampp.datasets.YCBVideoModels()
dataset = objslampp.datasets.YCBVideoDataset('train')
frame = dataset[0]

rgb = frame['color']
depth = frame['depth']
label = frame['label']
K = frame['meta']['intrinsic_matrix']
class_ids = frame['meta']['cls_indexes']

class_id = class_ids[0]
mask = label == class_id
cad_file = models.get_cad_model(class_id=class_id)
dim = 16
pitch = models.get_bbox_diagonal(cad_file) / dim

pcd = objslampp.geometry.pointcloud_from_depth(
    depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
)

pcd_ins = pcd[mask]
centroid = np.nanmean(pcd_ins, axis=0)
aabb_min = centroid - pitch * dim / 2
aabb_max = aabb_min + pitch * dim

camera = trimesh.scene.Camera(
    resolution=(rgb.shape[1], rgb.shape[0]),
    focal=(K[0, 0], K[1, 1]),
)

# -----------------------------------------------------------------------------

nonnan = ~np.isnan(depth)

octree = octomap.OcTree(pitch)
octree.insertPointCloud(
    pcd[mask & nonnan],
    np.array([0, 0, 0], dtype=float),
)

centers = trimesh.voxel.matrix_to_points(
    np.ones((dim, dim, dim), dtype=bool), pitch, aabb_min
)

leaves = walktree(octree)
sizes, coords, is_occupied = zip(*leaves)
sizes = np.array(list(sizes), dtype=float)
coords = np.array(list(coords), dtype=float)
is_occupied = np.array(list(is_occupied), dtype=bool)

sizes_occupied = sizes[is_occupied]
coords_occupied = coords[is_occupied]
sizes_empty = sizes[~is_occupied]
coords_empty = coords[~is_occupied]

LABEL_UNKNOWN = 0  # 4
LABEL_TARGET = 1
LABEL_UNTARGET = 2
LABEL_EMPTY = 7

labels = np.full((dim, dim, dim), LABEL_UNKNOWN, dtype=np.int32)

# occupied by target
kdtree = sklearn.neighbors.KDTree(coords_occupied)
dist, indices = kdtree.query(centers, k=1)
labels[dist[:, 0].reshape(dim, dim, dim) < pitch / 2] = LABEL_TARGET

# empty
kdtree = sklearn.neighbors.KDTree(coords_empty)
dist, indices = kdtree.query(centers, k=1)
labels[dist[:, 0].reshape(dim, dim, dim) < pitch / 2] = LABEL_EMPTY

# -----------------------------------------------------------------------------

octree = octomap.OcTree(pitch)
octree.insertPointCloud(
    pcd[~mask & nonnan],
    np.array([0, 0, 0], dtype=float),
)

leaves = walktree(octree)
sizes, coords, is_occupied = zip(*leaves)
sizes = np.array(list(sizes), dtype=float)
coords = np.array(list(coords), dtype=float)
is_occupied = np.array(list(is_occupied), dtype=bool)

sizes_occupied = sizes[is_occupied]
coords_occupied = coords[is_occupied]
sizes_empty = sizes[~is_occupied]
coords_empty = coords[~is_occupied]

# occupied by untarget
kdtree = sklearn.neighbors.KDTree(coords_occupied)
dist, indices = kdtree.query(centers, k=1)
labels[dist[:, 0].reshape(dim, dim, dim) < pitch / 2] = LABEL_UNTARGET

# empty
kdtree = sklearn.neighbors.KDTree(coords_empty)
dist, indices = kdtree.query(centers, k=1)
labels[dist[:, 0].reshape(dim, dim, dim) < pitch / 2] = LABEL_EMPTY

here = pathlib.Path(__file__).resolve().parent
(here / 'logs').mkdir(exist_ok=True)
npz_file = here / 'logs/occupancy_grid.npz'
T_cad2cam = np.vstack((frame['meta']['poses'][:, :, 0], [0, 0, 0, 1]))
np.savez_compressed(
    npz_file,
    labels=labels,
    pitch=pitch,
    origin=aabb_min,
    T_cad2cam=T_cad2cam,
)
print(f'==> Saved to {npz_file}')

# -----------------------------------------------------------------------------
# visualize occupancy grids

window = pyglet.window.Window(width=480 * 3, height=320)


@window.event
def on_key_press(symbol, modifiers):
    if modifiers == 0:
        if symbol == pyglet.window.key.Q:
            window.on_close()


gui = glooey.Gui(window)

hbox = glooey.HBox()
hbox.set_padding(5)

for i in range(2):
    scene = trimesh.Scene()

    # centers
    geom = trimesh.PointCloud(vertices=centers)
    # scene.add_geometry(geom)

    # bbox
    geom = trimesh.path.creation.box_outline(aabb_max - aabb_min)
    geom.apply_translation(centroid)
    scene.add_geometry(geom)

    # voxel
    if i == 0:
        caption = 'occupied (target or untarget)'
        matrix = np.isin(labels, [LABEL_TARGET, LABEL_UNTARGET])
    else:
        assert i == 1
        caption = 'empty'
        matrix = np.isin(labels, [LABEL_EMPTY])

    voxel = trimesh.voxel.Voxel(matrix, pitch, aabb_min)
    colors = imgviz.label2rgb(labels.reshape(1, -1)).reshape(dim, dim, dim, 3)
    alpha = np.full((dim, dim, dim, 1), 127, dtype=np.uint8)
    colors = np.concatenate((colors, alpha), axis=3)
    geom = voxel.as_boxes(colors=colors)
    scene.add_geometry(geom)

    vbox = glooey.VBox()
    vbox.add(glooey.text.Label(caption, color=(255, 255, 255)), size=0)
    vbox.add(trimesh.viewer.SceneWidget(scene))
    hbox.add(vbox)

# -----------------------------------------------------------------------------
# visualize point cloud

scene = trimesh.Scene()
# bbox
geom = trimesh.path.creation.box_outline(aabb_max - aabb_min)
geom.apply_translation(centroid)
scene.add_geometry(geom)
# point cloud
incube = np.greater(pcd, aabb_min, where=~np.isnan(pcd)).all(axis=2)
incube = incube & np.less(pcd, aabb_max, where=~np.isnan(pcd)).all(axis=2)
geom = trimesh.PointCloud(
    vertices=pcd[incube & nonnan], colors=rgb[incube & nonnan]
)
scene.add_geometry(geom)
vbox = glooey.VBox()
vbox.add(glooey.text.Label('point_cloud', color=(255, 255, 255)), size=0)
vbox.add(trimesh.viewer.SceneWidget(scene))
hbox.add(vbox)

# -----------------------------------------------------------------------------

gui.add(hbox)


def callback(dt, hbox, angles):
    scene_occupied = hbox.children[0].children[1].scene
    scene_empty = hbox.children[1].children[1].scene
    scene_pcd = hbox.children[2].children[1].scene

    scene_occupied.set_camera(angles=angles)
    scene_empty.set_camera(angles=angles)
    scene_pcd.set_camera(angles=angles)
    angles[1] += np.deg2rad(1)


angles = [0, 0, np.deg2rad(180)]
pyglet.clock.schedule_interval(callback, 1 / 30, hbox, angles)

pyglet.app.run()
