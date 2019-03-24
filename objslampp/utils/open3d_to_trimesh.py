import numpy as np
import open3d
import trimesh


def open3d_to_trimesh(src):
    if isinstance(src, open3d.TriangleMesh):
        dst = trimesh.Trimesh(
            vertices=np.asarray(src.vertices),
            faces=np.asarray(src.triangles),
            vertex_normals=np.asarray(src.vertex_normals),
        )
        dst.visual.vertex_colors = np.asarray(src.vertex_colors)
    else:
        raise ValueError('Unsupported type of src: {}'.format(type(src)))

    return dst
