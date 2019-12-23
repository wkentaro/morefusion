#!/usr/bin/env python

import argparse
import sys

import glooey
import imgviz
import numpy as np
import pybullet
import pyglet
import tqdm
import trimesh
import trimesh.transformations as tf
import trimesh.viewer

import morefusion

from common import Inference


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--refinement', choices=['icp', 'occupancy'], help='refinement'
)
args = parser.parse_args()

models = morefusion.datasets.YCBVideoModels()

inference = Inference(gpu=0)
frame, T_cad2cam_true, T_cad2cam_pred = inference(index=0, bg_class=True)
keep = frame['class_ids'] > 0
class_ids_fg = frame['class_ids'][keep]
instance_ids_fg = frame['instance_ids'][keep]

if args.refinement == 'icp':
    sys.path.insert(0, '../preliminary')
    from align_pointclouds import refinement  # NOQA

    registration = refinement(
        instance_ids=instance_ids_fg,
        class_ids=class_ids_fg,
        rgb=frame['rgb'],
        pcd=frame['pcd'],
        instance_label=frame['instance_label'],
        Ts_cad2cam_true=T_cad2cam_true,
        Ts_cad2cam_pred=T_cad2cam_pred,
    )
    T_cad2cam_pred = np.array([
        registration._Ts_cad2cam_pred[i] for i in instance_ids_fg
    ], dtype=float)
elif args.refinement == 'occupancy':
    sys.path.insert(0, '../preliminary')
    from align_occupancy_grids import refinement  # NOQA

    points_occupied = {}
    # ground
    dim = (64, 64, 16)
    pitch = 0.01
    matrix = np.ones(dim, dtype=bool)
    origin = - dim[0] * pitch / 2, - dim[1] * pitch / 2, - dim[2] * pitch
    points = trimesh.voxel.matrix_to_points(matrix, pitch, origin)
    points = trimesh.transform_points(
        points, frame['Ts_cad2cam'][frame['instance_ids'] == 0][0]
    )
    points_occupied[0] = points
    # bin
    mesh = trimesh.load(str(frame['cad_files'][1]))
    mesh.apply_transform(frame['Ts_cad2cam'][frame['instance_ids'] == 1][0])
    points_occupied[1] = mesh.voxelized(0.01).points

    keep = frame['class_ids'] > 0
    class_ids_fg = frame['class_ids'][keep]
    instance_ids_fg = frame['instance_ids'][keep]

    registration = refinement(
        instance_ids=instance_ids_fg,
        class_ids=class_ids_fg,
        rgb=frame['rgb'],
        pcd=frame['pcd'],
        instance_label=frame['instance_label'],
        Ts_cad2cam_true=T_cad2cam_true,
        Ts_cad2cam_pred=T_cad2cam_pred,
        points_occupied=points_occupied,
    )
    T_cad2cam_pred = np.array([
        registration._Ts_cad2cam_pred[i] for i in instance_ids_fg
    ], dtype=float)

T_cad2world_pred = frame['T_cam2world'] @ T_cad2cam_pred
T_cad2world_true = frame['T_cam2world'] @ T_cad2cam_true

# -----------------------------------------------------------------------------

window = pyglet.window.Window(width=640 * 2, height=480 * 2)


@window.event
def on_key_press(symbol, modifiers):
    if modifiers == 0:
        if symbol == pyglet.window.key.Q:
            window.on_close()


gui = glooey.Gui(window)
grid = glooey.Grid(num_rows=2, num_cols=2)
grid.set_padding(5)

K = frame['intrinsic_matrix']
height, width = frame['rgb'].shape[:2]
T_cam2world = frame['T_cam2world']
T_world2cam = np.linalg.inv(T_cam2world)

# -----------------------------------------------------------------------------
# rgb

image = morefusion.extra.pyglet.numpy_to_image(frame['rgb'])
widget = glooey.Image(image, responsive=True)
vbox = glooey.VBox()
vbox.add(glooey.Label(text='input rgb', color=(255, 255, 255)), size=0)
vbox.add(widget)
grid[0, 0] = vbox

# -----------------------------------------------------------------------------
# depth

scene = trimesh.Scene()

depth = frame['depth']
pcd = frame['pcd']
nonnan = ~np.isnan(depth)
# depth_viz = imgviz.depth2rgb(frame['depth'])
colormap = imgviz.label_colormap(value=200)
label_viz = imgviz.label2rgb(frame['instance_label'], colormap=colormap)
geom = trimesh.PointCloud(
    vertices=pcd[nonnan],
    colors=label_viz[nonnan],
)
scene.add_geometry(geom, transform=T_cam2world)

# -----------------------------------------------------------------------------
# cad

for i in range(T_cad2world_pred.shape[0]):
    class_id = class_ids_fg[i]
    cad = models.get_cad(class_id=class_id)
    cad.visual = cad.visual.to_color()
    # scene.add_geometry(cad, transform=T_cad2world_true[i])
    scene.add_geometry(cad, transform=T_cad2world_pred[i])

scene.camera.resolution = (width, height)
scene.camera.focal = (K[0, 0], K[1, 1])
scene.camera_transform = morefusion.extra.trimesh.to_opengl_transform(
    T_cam2world
)

widget = trimesh.viewer.SceneWidget(scene)
vbox = glooey.VBox()
vbox.add(glooey.Label(text='pcd & pred poses', color=(255, 255, 255)), size=0)
vbox.add(widget)
grid[0, 1] = vbox

# -----------------------------------------------------------------------------
# pybullet

morefusion.extra.pybullet.init_world(connection_method=pybullet.DIRECT)
# pybullet.resetDebugVisualizerCamera(
#     cameraDistance=0.8,
#     cameraYaw=30,
#     cameraPitch=-60,
#     cameraTargetPosition=(0, 0, 0),
# )

for ins_id, cad_file in frame['cad_files'].items():
    index = np.where(frame['instance_ids'] == ins_id)[0][0]
    T_cad2cam = frame['Ts_cad2cam'][index]
    T = frame['T_cam2world'] @ T_cad2cam
    morefusion.extra.pybullet.add_model(
        visual_file=cad_file,
        collision_file=morefusion.utils.get_collision_file(cad_file),
        position=tf.translation_from_matrix(T),
        orientation=tf.quaternion_from_matrix(T)[[1, 2, 3, 0]],
        base_mass=0,
    )

for i in range(T_cad2world_pred.shape[0]):
    class_id = class_ids_fg[i]
    visual_file = models.get_cad_file(class_id=class_id)
    collision_file = morefusion.utils.get_collision_file(visual_file)
    T = T_cad2world_pred[i]
    # T = T_cad2world_true[i]
    morefusion.extra.pybullet.add_model(
        visual_file=visual_file,
        collision_file=collision_file,
        position=tf.translation_from_matrix(T),
        orientation=tf.quaternion_from_matrix(T)[[1, 2, 3, 0]],
    )

rgbs_sim = []
for _ in tqdm.tqdm(range(60)):
    rgb, _, _ = morefusion.extra.pybullet.render_camera(
        T_cam2world, fovy=scene.camera.fov[1], height=height, width=width
    )
    rgbs_sim.append(rgb)
    pybullet.stepSimulation()

pybullet.disconnect()


image = morefusion.extra.pyglet.numpy_to_image(rgbs_sim[0])
widget = glooey.Image(image, responsive=True)
vbox = glooey.VBox()
vbox.add(glooey.Label(text='pred rendered', color=(255, 255, 255)), size=0)
vbox.add(widget)
grid[1, 0] = vbox


def callback(dt, image_widget):
    if image_widget.index >= len(rgbs_sim):
        image_widget.index = 0
    rgb = rgbs_sim[image_widget.index]
    image_widget.index += 1
    image = morefusion.extra.pyglet.numpy_to_image(rgb)
    image_widget.set_image(image)


image_widget = glooey.Image(responsive=True)
image_widget.index = 0
pyglet.clock.schedule_interval(callback, 1 / 30, image_widget)
vbox = glooey.VBox()
vbox.add(
    glooey.Label(text='pred poses then gravity', color=(255, 255, 255)),
    size=0,
)
vbox.add(image_widget)
grid[1, 1] = vbox

# -----------------------------------------------------------------------------

gui.add(grid)
pyglet.app.run()
