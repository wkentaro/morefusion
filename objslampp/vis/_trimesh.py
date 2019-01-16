import copy
import math
import typing

import trimesh


def box_to_wired_box(
    box: trimesh.primitives.Box
) -> typing.List[trimesh.base.Geometry]:
    '''Returns wired box for the given box.'''
    indices = [0, 1, 3, 2, 0, 4, 5, 7, 6, 4, 0, 2, 6, 7, 3, 1, 5]
    return trimesh.load_path(box.vertices[indices])


def wired_box(extents, transform=None, translation=None):
    box = trimesh.creation.box(extents=extents)
    if transform is not None:
        assert translation is None
        box.apply_transform(transform)
    if translation is not None:
        assert transform is None
        box.apply_translation(translation)
    return box_to_wired_box(box.bounding_box)


def _get_tile_shape(num):
    x_num = int(math.sqrt(num))  # floor
    y_num = 0
    while x_num * y_num < num:
        y_num += 1
    return x_num, y_num


def tile_meshes(meshes, shape=None):
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
