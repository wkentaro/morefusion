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
