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


def display_scenes(
    data,
    height=480,
    width=640,
    tile=None,
    caption=None,
    rotation_scaling=1,
):
    import glooey

    scenes = None
    scenes_group = None
    if isinstance(data, types.GeneratorType):
        next_data = next(data)
        if isinstance(next_data, types.GeneratorType):
            scenes = next(next_data)
            scenes_group = next_data
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
    HEIGHT_LABEL_WIDGET = 19
    PADDING_GRID = 1
    for config in configs:
        try:
            window = pyglet.window.Window(
                height=(height + HEIGHT_LABEL_WIDGET) * nrow,
                width=(width + PADDING_GRID * 2) * ncol,
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

    def usage():
        return '''\
Usage:
  q: quit
  s: play / pause
  z: reset view
  n: next
  r: rotate view (clockwise)
  R: rotate view (anti-clockwise)\
'''

    @window.event
    def on_key_press(symbol, modifiers):
        if symbol == pyglet.window.key.Q:
            window.on_close()
        elif window.scenes_group and symbol == pyglet.window.key.S:
            window.play = not window.play
        elif symbol == pyglet.window.key.Z:
            for name in scenes:
                if isinstance(widgets[name], trimesh.viewer.SceneWidget):
                    widgets[name].reset_view()
        elif symbol == pyglet.window.key.N:
            window.next = True
        elif symbol == pyglet.window.key.R:
            # rotate camera
            window.rotate = not window.rotate  # 0/1
            if modifiers == pyglet.window.key.MOD_SHIFT:
                window.rotate *= -1
        elif symbol == pyglet.window.key.H:
            print(usage())

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
                        np.deg2rad(window.rotate * rotation_scaling),
                        axis,
                        point=scene.centroid,
                    ) @ camera.transform
                    widget.view['ball']._n_pose = camera.transform
            return

        if window.scenes_group and (window.next or window.play):
            try:
                scenes = next(window.scenes_group)
                for key, widget in widgets.items():
                    scene = scenes[key]
                    if isinstance(widget, trimesh.viewer.SceneWidget):
                        assert isinstance(scene, trimesh.Scene)
                        widget.scene.geometry = scene.geometry
                        widget.scene.graph = scene.graph
                        widget.view['ball']._n_pose = scene.camera.transform
                        widget._draw()
                    elif isinstance(widget, glooey.Image):
                        widget.set_image(numpy_to_image(scene))
            except StopIteration:
                print('Reached the end of the scenes')
                window.play = False
            window.next = False

    gui = glooey.Gui(window)
    grid = glooey.Grid()
    grid.set_padding(PADDING_GRID)

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
