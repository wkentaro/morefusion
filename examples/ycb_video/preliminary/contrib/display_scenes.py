import math

import glooey
import numpy as np
import pyglet
import trimesh
import trimesh.transformations as tf


def _get_tile_shape(num, hw_ratio=1):
    r_num = int(round(math.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def display_scenes(scenes, height=480, width=640, tile=None, caption=None):
    if tile is None:
        nrow, ncol = _get_tile_shape(len(scenes), hw_ratio=height / width)
    else:
        nrow, ncol = tile

    cameras = {}
    for name, scene in scenes.items():
        cameras[name] = scene.camera.copy()

    window = pyglet.window.Window(
        height=height * nrow,
        width=width * ncol,
        caption=caption,
    )
    window.rotate = 0

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()
            elif symbol == pyglet.window.key.Z:
                for name in scenes:
                    widgets[name].reset_view()
        if symbol == pyglet.window.key.R:
            # rotate camera
            window.rotate = not window.rotate  # 0/1
            if modifiers == pyglet.window.key.MOD_SHIFT:
                window.rotate *= -1

    def callback(dt):
        if window.rotate:
            for widget in widgets.values():
                scene = widget.scene
                camera = scene.camera
                axis = tf.transform_points(
                    [[0, 1, 0]], camera.transform, translate=False
                )[0]
                camera.transform = tf.rotation_matrix(
                    np.deg2rad(window.rotate), axis, point=scene.centroid
                ) @ camera.transform
                widget.view['ball']._n_pose = camera.transform
            return

    gui = glooey.Gui(window)
    grid = glooey.Grid()
    grid.set_padding(5)

    widgets = {}
    for i, (name, scene) in enumerate(scenes.items()):
        vbox = glooey.VBox()
        vbox.add(glooey.Label(text=name, color=(255,) * 3), size=0)
        widgets[name] = trimesh.viewer.SceneWidget(scene)
        vbox.add(widgets[name])
        grid[i // ncol, i % ncol] = vbox

    gui.add(grid)

    pyglet.clock.schedule_interval(callback, 1 / 30)
    pyglet.app.run()
    pyglet.clock.unschedule(callback)
