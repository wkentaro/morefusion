import copy
import math
import typing

import numpy as np
import trimesh
import trimesh.viewer


def box_to_wired_box(
    box: trimesh.primitives.Box
) -> typing.List[trimesh.base.Geometry]:
    '''Returns wired box for the given box.'''
    indices = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 0, 2, 6, 7, 3, 1, 5]
    return trimesh.load_path(box.vertices[indices])


def wired_box(
    extents: typing.Union[tuple, list, np.ndarray],
    transform: typing.Optional[np.ndarray] = None,
    translation: typing.Optional[np.ndarray] = None,
) -> typing.List[typing.Any]:
    box = trimesh.creation.box(extents=extents)
    if transform is not None:
        assert translation is None
        box.apply_transform(transform)
    if translation is not None:
        assert transform is None
        box.apply_translation(translation)
    return box_to_wired_box(box.bounding_box)


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
    if transform is None:
        transform = np.eye(4)
    return transform @ trimesh.transformations.rotation_matrix(
        np.deg2rad(-180), [1, 0, 0]
    )
