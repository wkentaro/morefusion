import numpy as np
import trimesh


def box_outline_from_voxel_grid(vg):
    assert isinstance(vg, trimesh.voxel.VoxelGrid)
    geom = trimesh.path.creation.box_outline(np.array(vg.shape) * vg.scale)
    geom.apply_translation((np.array(vg.shape) / 2.0 - 0.5) * vg.scale)
    geom.apply_translation(vg.origin)
    return geom
