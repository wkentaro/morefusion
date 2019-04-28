#!/usr/bin/env python

import io
import json
import pathlib

import chainer
from chainer.backends import cuda
import glooey
import imgviz
import numpy as np
import PIL.Image
import pybullet
import pyglet
import tqdm
import trimesh
import trimesh.viewer
import trimesh.transformations as tf

import objslampp

import contrib
import synthetic_data


gpu = 0
log_dir = pathlib.Path('./logs.20190417.cad_only/20190412_142459.904281')
args_file = log_dir / 'args'
model_file = log_dir / 'snapshot_model_best_auc_add.npz'
root_dir = '~/data/datasets/wkentaro/objslampp/ycb_video/synthetic_data/20190428_165745.028250'  # NOQA
root_dir = pathlib.Path(root_dir).expanduser()
class_ids = [2]

# -----------------------------------------------------------------------------

with open(args_file) as f:
    args_data = json.load(f)

model = contrib.models.BaselineModel(
    freeze_until=args_data['freeze_until'],
    voxelization=args_data.get('voxelization', 'average'),
)
if gpu >= 0:
    cuda.get_device_from_id(gpu).use()
    model.to_gpu()
chainer.serializers.load_npz(model_file, model)

dataset = contrib.datasets.BinTypeDataset(
    root_dir=root_dir,
    class_ids=class_ids,
)

models = objslampp.datasets.YCBVideoModels()

# -----------------------------------------------------------------------------

index = 0
frame = dataset.get_frame(index)
with chainer.using_config('debug', True):
    examples = dataset.get_examples(index)

inputs = chainer.dataset.concat_examples(examples, device=gpu)

with chainer.no_backprop_mode() and \
        chainer.using_config('train', False):
    quaternion_pred, translation_pred = model.predict(
        class_id=inputs['class_id'],
        pitch=inputs['pitch'],
        rgb=inputs['rgb'],
        pcd=inputs['pcd'],
    )

quaternion_pred = cuda.to_cpu(quaternion_pred.array)
translation_pred = cuda.to_cpu(translation_pred.array)
quaternion_true = cuda.to_cpu(inputs['quaternion_true'])
translation_true = cuda.to_cpu(inputs['translation_true'])

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

with io.BytesIO() as f:
    PIL.Image.fromarray(frame['rgb']).save(f, format='JPEG')
    image = pyglet.image.load(filename=None, file=f)
widget = glooey.Image(image, responsive=True)
vbox = glooey.VBox()
vbox.add(glooey.Label(text='input rgb', color=(255, 255, 255)), size=0)
vbox.add(widget)
grid[0, 0] = vbox

# -----------------------------------------------------------------------------
# depth

scene = trimesh.Scene()

depth = frame['depth']
pcd = objslampp.geometry.pointcloud_from_depth(
    depth, fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2]
)
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

batch_size = len(examples)
T_cad2world_pred = np.zeros((batch_size, 4, 4), dtype=float)
T_cad2world_true = np.zeros((batch_size, 4, 4), dtype=float)
for i in range(batch_size):
    T_cad2cam_pred = tf.quaternion_matrix(quaternion_pred[i])
    T_cad2cam_pred[:3, 3] = translation_pred[i]
    T_cad2world_pred[i] = frame['T_cam2world'] @ T_cad2cam_pred

    T_cad2cam_true = tf.quaternion_matrix(quaternion_true[i])
    T_cad2cam_true[:3, 3] = translation_true[i]
    T_cad2world_true[i] = frame['T_cam2world'] @ T_cad2cam_true

for i in range(batch_size):
    class_id = int(inputs['class_id'][i])
    cad_file = models.get_cad_model(class_id=class_id)
    cad = trimesh.load(str(cad_file))
    cad.visual = cad.visual.to_color()
    # scene.add_geometry(cad, transform=T_cad2world_true[i])
    scene.add_geometry(cad, transform=T_cad2world_pred[i])

scene.camera.resolution = (width, height)
scene.camera.focal = (K[0, 0], K[1, 1])
scene.camera.transform = objslampp.extra.trimesh.camera_transform(T_cam2world)

widget = trimesh.viewer.SceneWidget(scene)
vbox = glooey.VBox()
vbox.add(glooey.Label(text='pcd & pred poses', color=(255, 255, 255)), size=0)
vbox.add(widget)
grid[0, 1] = vbox

# -----------------------------------------------------------------------------
# pybullet

objslampp.extra.pybullet.init_world(connection_method=pybullet.DIRECT)
# pybullet.resetDebugVisualizerCamera(
#     cameraDistance=0.8,
#     cameraYaw=30,
#     cameraPitch=-60,
#     cameraTargetPosition=(0, 0, 0),
# )

for i in range(len(examples)):
    class_id = int(inputs['class_id'][i])
    visual_file = models.get_cad_model(class_id=class_id)
    collision_file = synthetic_data.simulation\
        .scene_generation.base.SceneGenerationBase\
        ._get_collision_file(visual_file)
    T = T_cad2world_pred[i]
    # T = T_cad2world_true[i]
    objslampp.extra.pybullet.add_model(
        visual_file=visual_file,
        collision_file=collision_file,
        position=tf.translation_from_matrix(T),
        orientation=tf.quaternion_from_matrix(T)[[1, 2, 3, 0]],
    )

rgbs_sim = []
for _ in tqdm.tqdm(range(60)):
    rgb, _, _ = objslampp.extra.pybullet.render_camera(
        T_cam2world, fovy=scene.camera.fov[1], height=height, width=width
    )
    rgbs_sim.append(rgb)
    pybullet.stepSimulation()

pybullet.disconnect()


with io.BytesIO() as f:
    PIL.Image.fromarray(rgbs_sim[0]).save(f, format='JPEG')
    image = pyglet.image.load(filename=None, file=f)
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
    with io.BytesIO() as f:
        PIL.Image.fromarray(rgb).save(f, format='JPEG')
        image = pyglet.image.load(filename=None, file=f)
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
