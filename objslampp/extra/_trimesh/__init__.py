import copy
import io
import math
import typing
import warnings

import numpy as np
import PIL.Image
import trimesh
import trimesh.viewer


def _get_tile_shape(num: int) -> typing.Tuple[int, int]:
    x_num = int(math.sqrt(num))  # floor
    y_num = 0
    while x_num * y_num < num:
        y_num += 1
    return x_num, y_num


def tile_meshes(
    meshes: typing.List[typing.Any],
    shape: typing.Optional[tuple] = None,
) -> trimesh.Scene:
    meshes = copy.deepcopy(meshes)

    if shape is None:
        shape = _get_tile_shape(len(meshes))

    transforms = []
    min_z = float('inf')
    for i, mesh in enumerate(meshes):
        scale = mesh.bounding_box.extents.max()
        mesh.apply_scale(0.5 / scale)

        min_z = min(min_z, mesh.vertices.min())

        row = i // shape[1]
        col = i % shape[1]
        transform = trimesh.transformations.translation_matrix([row, col, 0])
        transforms.append(transform)

    scene = trimesh.Scene()

    for mesh, transform in zip(meshes, transforms):
        scene.add_geometry(mesh, transform=transform)

    plane_depth = 0.01
    plane = trimesh.creation.box([shape[0], shape[1], plane_depth])
    plane.apply_translation([
        shape[0] / 2. - 0.5,
        shape[1] / 2. - 0.5,
        - plane_depth / 2. + min_z,
    ])
    plane.visual.face_colors = [[1., 1., 1.]] * len(plane.faces)
    scene.add_geometry(plane)

    return scene


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
