import io

import numpy as np
import PIL.Image
import trimesh
import trimesh.viewer


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

    wall_xp = trimesh.creation.box((thickness, ylength, zlength))
    wall_xn = wall_xp.copy()
    wall_xp.apply_translation((xlength / 2, 0, 0))
    wall_xn.apply_translation((-xlength / 2, 0, 0))
    mesh = wall_xp.copy()
    mesh += wall_xn

    wall_yp = trimesh.creation.box((xlength, thickness, zlength))
    wall_yn = wall_yp.copy()
    wall_yp.apply_translation((0, ylength / 2 - thickness / 2, 0))
    wall_yn.apply_translation((0, -ylength / 2 + thickness / 2, 0))
    mesh += wall_yp
    mesh += wall_yn

    wall_zn = trimesh.creation.box((xlength, ylength, thickness))
    wall_zn.apply_translation((0, 0, -zlength / 2 + thickness / 2))
    mesh += wall_zn

    if color is None:
        color = (1.0, 1.0, 1.0)
    mesh.visual.face_colors = color

    return mesh
