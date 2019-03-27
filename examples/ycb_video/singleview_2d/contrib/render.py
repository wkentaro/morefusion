import numpy as np
import trimesh.transformations as tf

import objslampp


def render(cad_file, T_cad2cam, K, height, width):
    import pybullet

    pybullet.connect(pybullet.DIRECT)
    objslampp.extra.pybullet.add_model(
        cad_file,
        position=tf.translation_from_matrix(T_cad2cam),
        orientation=tf.quaternion_from_matrix(T_cad2cam)[[1, 2, 3, 0]],
        register=False
    )

    fovy = np.rad2deg(2 * np.arctan(height / (2 * K[1, 1])))
    projection_matrix = pybullet.computeProjectionMatrixFOV(
        fov=fovy, aspect=1. * width / height, farVal=1000., nearVal=0.01
    )

    view_matrix = pybullet.computeViewMatrix(
        cameraEyePosition=[0, 0, 0],
        cameraTargetPosition=[0, 0, 1],
        cameraUpVector=[0, -1, 0],
    )
    _, _, rgb, _, segm = pybullet.getCameraImage(
        width,
        height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
    )
    rgb = rgb[:, :, :3]
    mask = (segm == 0).astype(np.int32)
    pybullet.disconnect()
    return rgb, mask
