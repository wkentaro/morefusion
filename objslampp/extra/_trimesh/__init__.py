import io
import warnings

import numpy as np
import PIL.Image
import trimesh
import trimesh.viewer

from .display_scenes import display_scenes  # NOQA
from .tile_meshes import tile_meshes  # NOQA


def show_with_rotation(scene, step=None, init_angles=None, **kwargs):
    if step is None:
        step = (0, np.deg2rad(1), 0)
    if init_angles is None:
        init_angles = (0, 0, 0)

    step = np.asarray(step, dtype=float)
    init_angles = np.asarray(init_angles, dtype=float)

    def callback(scene):
        if hasattr(scene, 'angles'):
            scene.angles += step
        else:
            scene.angles = init_angles
        scene.set_camera(angles=scene.angles)

    return trimesh.viewer.SceneViewer(scene=scene, callback=callback, **kwargs)


def camera_transform(transform=None):
    warnings.warn("camera_transform is deprecated, use to_opengl_transform",
                  DeprecationWarning)
    return to_opengl_transform(transform=transform)


def to_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )


def from_opengl_transform(transform=None):
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(180), [1, 0, 0]
    )


def save_image(scene, **kwargs):
    with io.BytesIO() as f:
        data = scene.save_image(resolution=scene.camera.resolution, **kwargs)
        f.write(data)
        return np.asarray(PIL.Image.open(f))


def bin_model(extents, thickness, color=None):
    xlength, ylength, zlength = extents

    mesh = trimesh.Trimesh()

    wall_xp = trimesh.creation.box((thickness, ylength, zlength))
    wall_xn = wall_xp.copy()
    wall_xp.apply_translation((xlength / 2, 0, 0))
    wall_xn.apply_translation((- xlength / 2, 0, 0))
    mesh += wall_xp
    mesh += wall_xn

    wall_yp = trimesh.creation.box((xlength, thickness, zlength))
    wall_yn = wall_yp.copy()
    wall_yp.apply_translation((0, ylength / 2 - thickness / 2, 0))
    wall_yn.apply_translation((0, - ylength / 2 + thickness / 2, 0))
    mesh += wall_yp
    mesh += wall_yn

    wall_zn = trimesh.creation.box((xlength, ylength, thickness))
    wall_zn.apply_translation((0, 0, - zlength / 2 + thickness / 2))
    mesh += wall_zn

    if color is None:
        color = (1.0, 1.0, 1.0)
    mesh.visual.face_colors = color

    return mesh
