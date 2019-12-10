import numpy as np
import trimesh


def box_outline_from_voxel(voxel):
    assert isinstance(voxel, trimesh.voxel.Voxel)
    geom = trimesh.path.creation.box_outline(
        np.array(voxel.shape) * voxel.pitch
    )
    geom.apply_translation(voxel.origin)
    geom.apply_translation((np.array(voxel.shape) / 2.0 - 0.5) * voxel.pitch)
    return geom
