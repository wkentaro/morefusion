import math
import types

import numpy as np
import pyglet
import trimesh
import trimesh.transformations as tf

from .._pyglet import numpy_to_image


def _get_tile_shape(num, hw_ratio=1):
    r_num = int(round(math.sqrt(num / hw_ratio)))  # weighted by wh_ratio
    c_num = 0
    while r_num * c_num < num:
        c_num += 1
    while (r_num - 1) * c_num >= num:
        r_num -= 1
    return r_num, c_num


def display_scenes(data, height=480, width=640, tile=None, caption=None):
    import glooey

    scenes = None
    scenes_group = None
    scenes_ggroup = None
    if isinstance(data, types.GeneratorType):
        next_data = next(data)
        if isinstance(next_data, types.GeneratorType):
            scenes = next(next_data)
            scenes_group = next_data
            scenes_ggroup = data
        else:
            scenes = next_data
            scenes_group = data
    else:
        scenes = data

    if tile is None:
        nrow, ncol = _get_tile_shape(len(scenes), hw_ratio=height / width)
    else:
        nrow, ncol = tile

    configs = [
        pyglet.gl.Config(
            sample_buffers=1, samples=4, depth_size=24, double_buffer=True
        ),
        pyglet.gl.Config(double_buffer=True),
    ]
    for config in configs:
        try:
            window = pyglet.window.Window(
                height=height * nrow,
                width=width * ncol,
                caption=caption,
                config=config,
            )
            break
        except pyglet.window.NoSuchConfigException:
            pass
    window.rotate = 0

    if scenes_group:
        window.play = False
        window.next = False
    window.scenes_group = scenes_group
    window.scenes_ggroup = scenes_ggroup

    @window.event
    def on_key_press(symbol, modifiers):
        if modifiers == 0:
            if symbol == pyglet.window.key.Q:
                window.on_close()
            elif window.scenes_group and symbol == pyglet.window.key.S:
                window.play = not window.play
            elif symbol == pyglet.window.key.Z:
                for name in scenes:
                    widgets[name].reset_view()
        if symbol == pyglet.window.key.N:
            if modifiers == 0:
                window.next = True
            elif window.scenes_ggroup and \
                    modifiers == pyglet.window.key.MOD_SHIFT:
                try:
                    window.scenes_group = next(window.scenes_ggroup)
                    window.next = True
                except StopIteration:
                    return
        if symbol == pyglet.window.key.R:
            # rotate camera
            window.rotate = not window.rotate  # 0/1
            if modifiers == pyglet.window.key.MOD_SHIFT:
                window.rotate *= -1

    def callback(dt):
        if window.rotate:
            for widget in widgets.values():
                if isinstance(widget, trimesh.viewer.SceneWidget):
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

        if window.scenes_group and (window.next or window.play):
            try:
                scenes = next(window.scenes_group)
                for key, widget in widgets.items():
                    if isinstance(widget, trimesh.viewer.SceneWidget):
                        widget.scene.geometry.update(scenes[key].geometry)
                        widget.scene.graph.load(
                            scenes[key].graph.to_edgelist()
                        )
                        widget._draw()
                    elif isinstance(widget, glooey.Image):
                        widget.set_image(numpy_to_image(scenes[key]))
            except StopIteration:
                window.play = False
            window.next = False

    gui = glooey.Gui(window)
    grid = glooey.Grid()
    grid.set_padding(5)

    widgets = {}
    trackball = None
    for i, (name, scene) in enumerate(scenes.items()):
        vbox = glooey.VBox()
        vbox.add(glooey.Label(text=name, color=(255,) * 3), size=0)
        if isinstance(scene, trimesh.Scene):
            widgets[name] = trimesh.viewer.SceneWidget(scene)
            if trackball is None:
                trackball = widgets[name].view['ball']
            else:
                widgets[name].view['ball'] = trackball
        elif isinstance(scene, np.ndarray):
            widgets[name] = glooey.Image(
                numpy_to_image(scene), responsive=True
            )
        else:
            raise TypeError(f'unsupported type of scene: {scene}')
        vbox.add(widgets[name])
        grid[i // ncol, i % ncol] = vbox

    gui.add(grid)

    pyglet.clock.schedule_interval(callback, 1 / 30)
    pyglet.app.run()
    pyglet.clock.unschedule(callback)
