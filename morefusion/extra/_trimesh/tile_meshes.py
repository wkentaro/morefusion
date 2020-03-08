import copy
import math
import typing

import trimesh
import trimesh.viewer


def _get_tile_shape(num: int) -> typing.Tuple[int, int]:
    x_num = int(math.sqrt(num))  # floor
    y_num = 0
    while x_num * y_num < num:
        y_num += 1
    return x_num, y_num


def tile_meshes(
    meshes: typing.List[typing.Any], shape: typing.Optional[tuple] = None,
) -> trimesh.Scene:
    meshes = copy.deepcopy(meshes)

    if shape is None:
        shape = _get_tile_shape(len(meshes))

    transforms = []
    min_z = float("inf")
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
    plane.apply_translation(
        [
            shape[0] / 2.0 - 0.5,
            shape[1] / 2.0 - 0.5,
            -plane_depth / 2.0 + min_z,
        ]
    )
    plane.visual.face_colors = [[1.0, 1.0, 1.0]] * len(plane.faces)
    scene.add_geometry(plane)

    return scene
