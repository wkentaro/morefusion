import numpy as np
import open3d
import trimesh


def trimesh_to_open3d(src):
    if isinstance(src, trimesh.Trimesh):
        dst = open3d.TriangleMesh()
        dst.vertices = open3d.Vector3dVector(src.vertices)
        dst.triangles = open3d.Vector3iVector(src.faces)
        vertex_colors = (
            src.visual.vertex_colors[:, :3].astype(np.float32) / 255
        )
        dst.vertex_colors = open3d.Vector3dVector(vertex_colors)
        dst.compute_vertex_normals()
    elif isinstance(src, trimesh.PointCloud):
        dst = open3d.PointCloud()
        dst.points = open3d.Vector3dVector(src.vertices)
        if src.colors:
            colors = src.colors
            colors = (colors[:, :3] / 255.0).astype(float)
            dst.colors = open3d.Vector3dVector(colors)
    elif isinstance(src, trimesh.scene.Camera):
        dst = open3d.PinholeCameraIntrinsic(
            width=src.resolution[0],
            height=src.resolution[1],
            fx=src.K[0, 0],
            fy=src.K[1, 1],
            cx=src.K[0, 2],
            cy=src.K[1, 2],
        )
    elif isinstance(src, trimesh.path.Path3D):
        lines = []
        for entity in src.entities:
            for i, j in zip(entity.points[:-1], entity.points[1:]):
                lines.append((i, j))
        lines = np.vstack(lines)
        points = src.vertices
        dst = open3d.LineSet()
        dst.lines = open3d.Vector2iVector(lines)
        dst.points = open3d.Vector3dVector(points)
    elif isinstance(src, list):
        dst = [trimesh_to_open3d(x) for x in src]
    else:
        raise ValueError("Unsupported type of src: {}".format(type(src)))

    return dst
