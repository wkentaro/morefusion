import typing

import numpy as np
import trimesh

from .. import geometry


unique_ids: list = []


def init_world(connection_method=None) -> None:
    import pybullet
    import pybullet_data

    if connection_method is None:
        connection_method = pybullet.GUI
    pybullet.connect(connection_method)
    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())

    pybullet.loadURDF("plane.urdf")
    pybullet.setGravity(0, 0, -9.8)


def del_world() -> None:
    import pybullet

    pybullet.disconnect()

    global unique_ids
    unique_ids = []


def get_debug_visualizer_image() -> typing.Tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    import pybullet

    width, height, *_ = pybullet.getDebugVisualizerCamera()
    _, _, rgba, depth, segm = pybullet.getCameraImage(
        width=width, height=height
    )
    rgb = rgba[:, :, :3]
    depth[segm == -1] = np.nan
    return rgb, depth, segm


def add_model(
    visual_file: str,
    collision_file: typing.Optional[str] = None,
    position: typing.Optional[typing.Sequence] = None,
    orientation: typing.Optional[typing.Sequence] = None,
    mesh_scale: typing.Optional[
        typing.Union[int, float, typing.Sequence]
    ] = None,
    register: bool = True,
    base_mass=1,
) -> int:
    import pybullet

    visual_file = str(visual_file)
    if collision_file is None:
        collision_file = visual_file
    collision_file = str(collision_file)

    if position is None:
        position = [0, 0, 0]
    if orientation is None:
        orientation = [0, 0, 0, 1]
    if mesh_scale is None:
        mesh_scale = [1, 1, 1]
    if isinstance(mesh_scale, (int, float)):
        mesh_scale = [mesh_scale] * 3

    visual_shape_id = pybullet.createVisualShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=visual_file,
        visualFramePosition=(0, 0, 0),
        meshScale=mesh_scale,
    )
    collision_shape_id = pybullet.createCollisionShape(
        shapeType=pybullet.GEOM_MESH,
        fileName=collision_file,
        collisionFramePosition=(0, 0, 0),
        meshScale=mesh_scale,
    )
    unique_id = pybullet.createMultiBody(
        baseMass=base_mass,
        baseInertialFramePosition=(0, 0, 0),
        baseCollisionShapeIndex=collision_shape_id,
        baseVisualShapeIndex=visual_shape_id,
        basePosition=position,
        baseOrientation=orientation,
        useMaximalCoordinates=False,
    )
    if register:
        unique_ids.append(unique_id)
    return unique_id


def shape_id_to_str(shape_id: int) -> str:
    import pybullet

    id_to_str = {
        pybullet.GEOM_BOX: "GEOM_BOX",
        pybullet.GEOM_CAPSULE: "GEOM_CAPSULE",
        pybullet.GEOM_CYLINDER: "GEOM_CYLINDER",
        pybullet.GEOM_MESH: "GEOM_MESH",
        pybullet.GEOM_PLANE: "GEOM_PLANE",
        pybullet.GEOM_SPHERE: "GEOM_SPHERE",
    }
    return id_to_str[shape_id]


def get_trimesh_scene(axis: bool = False, bbox: bool = False) -> trimesh.Scene:
    """Returns trimesh scene."""
    import pybullet

    scene = trimesh.Scene()
    for unique_id in unique_ids:
        _, _, shape_id, _, mesh_file, *_ = pybullet.getVisualShapeData(
            unique_id
        )[0]
        mesh_file = mesh_file.decode()
        if pybullet.GEOM_MESH != shape_id:
            raise ValueError(
                f"Unsupported shape_id: {shape_id_to_str(shape_id)}"
            )

        pos, ori = pybullet.getBasePositionAndOrientation(unique_id)
        t = np.array(pos, dtype=float)
        R = pybullet.getMatrixFromQuaternion(ori)
        R = np.array(R, dtype=float).reshape(3, 3)
        transform = geometry.compose_transform(R=R, t=t)

        mesh = trimesh.load_mesh(mesh_file)
        scene.add_geometry(
            mesh, node_name=str(unique_id), transform=transform,
        )

        if bbox:
            scene.add_geometry(
                trimesh.path.creation.box_outline(mesh.bounding_box),
                transform=transform,
            )

        if axis:
            origin_size = np.max(mesh.bounding_box.extents) * 0.05
            scene.add_geometry(
                trimesh.creation.axis(origin_size), transform=transform,
            )
    return scene


def aabb_contained_ratio(aabb1, aabb2) -> float:
    """Returns how much aabb2 is contained by aabb1."""
    import pybullet

    if isinstance(aabb1, int):
        aabb1 = pybullet.getAABB(aabb1)
    if isinstance(aabb2, int):
        aabb2 = pybullet.getAABB(aabb2)

    aabb1_min, aabb1_max = aabb1
    aabb1_min = np.array(aabb1_min)
    aabb1_max = np.array(aabb1_max)

    aabb2_min, aabb2_max = aabb2
    aabb2_min = np.array(aabb2_min)
    aabb2_max = np.array(aabb2_max)

    def get_volume(aabb_min, aabb_max):
        aabb_extents = aabb_max - aabb_min
        if np.any(aabb_extents <= 0):
            return 0
        return np.prod(aabb_extents)

    volume_intersect = get_volume(
        np.maximum(aabb1_min, aabb2_min), np.minimum(aabb1_max, aabb2_max),
    )
    volume2 = get_volume(aabb2_min, aabb2_max)
    # volume1 = get_volume(aabb1_min, aabb1_max)
    # iou = volume_intersect / (volume1 + volume2 - volume_intersect)
    ratio = volume_intersect / volume2
    if ratio < 0:
        ratio = 0
    return ratio


def get_top_image(visual_file: str) -> np.ndarray:
    import pybullet

    pybullet.connect(pybullet.DIRECT)

    add_model(visual_file=visual_file, register=False)

    view_matrix = pybullet.computeViewMatrix(
        cameraEyePosition=[0.15, 0.15, 0.15],
        cameraTargetPosition=[0, 0, 0],
        cameraUpVector=[0, -1, 0],
    )
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=60, aspect=1, nearVal=0.01, farVal=100,
    )
    H, W, rgba, *_ = pybullet.getCameraImage(
        256, 256, viewMatrix=view_matrix, projectionMatrix=projection_matrix
    )

    pybullet.disconnect()

    rgba = np.asarray(rgba, dtype=np.uint8).reshape(H, W, 4)
    rgb = rgba[:, :, :3]

    return rgb


def get_camera_image(view_matrix, fovy, height, width):
    import pybullet

    far = 1000
    near = 0.01
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=fovy, aspect=1.0 * width / height, farVal=far, nearVal=near
    )
    _, _, rgba, depth, segm = pybullet.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
    )
    rgb = rgba[:, :, :3]
    depth = np.asarray(depth, dtype=np.float32).reshape(height, width)
    depth = far * near / (far - (far - near) * depth)
    depth[segm == -1] = np.nan
    return rgb, depth, segm


def render_camera(T_cam2world, fovy, height, width):
    view_matrix = T_cam2world.copy()
    view_matrix[:3, 3] = 0
    view_matrix[3, :3] = np.linalg.inv(T_cam2world)[:3, 3]
    view_matrix[:, 1] *= -1
    view_matrix[:, 2] *= -1
    view_matrix = view_matrix.flatten()
    rgb, depth, segm = get_camera_image(
        view_matrix=view_matrix, fovy=fovy, height=height, width=width,
    )
    return rgb, depth, segm


def render_cad(visual_file, Ts_cad2cam, fovy, height, width, scale=None):
    assert isinstance(visual_file, str)

    Ts_cad2cam = np.asarray(Ts_cad2cam)
    ndim = Ts_cad2cam.ndim
    if ndim == 2:
        Ts_cad2cam = Ts_cad2cam[None]
    assert Ts_cad2cam.shape == (Ts_cad2cam.shape[0], 4, 4)

    import pybullet

    pybullet.connect(pybullet.DIRECT)

    add_model(visual_file, mesh_scale=scale, register=False)

    rgbs = []
    depths = []
    masks = []
    for T_cad2cam in Ts_cad2cam:
        T_cam2cad = np.linalg.inv(T_cad2cam)
        rgb, depth, segm = render_camera(
            T_cam2world=T_cam2cad, fovy=fovy, height=height, width=width
        )
        rgbs.append(rgb)
        depths.append(depth)
        masks.append(segm == 0)
    rgbs = np.asarray(rgbs)
    depths = np.asarray(depths)
    masks = np.asarray(masks)

    pybullet.disconnect()

    if ndim == 2:
        assert len(rgbs) == len(depths) == len(masks) == 1
        rgbs = rgbs[0]
        depths = depths[0]
        masks = masks[0]

    return rgbs, depths, masks
