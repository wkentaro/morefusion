#!/usr/bin/env python

import path
import tempfile

import numpy as np
import pybullet
import trimesh

import objslampp

import cv_bridge
import rospy
from ros_objslampp_srvs.srv import RenderVoxelGridArray
from ros_objslampp_srvs.srv import RenderVoxelGridArrayResponse

from voxel_grids_to_mesh_markers import grid_msg_to_mesh


class RenderVoxelGrids:

    def __init__(self):
        super().__init__()
        self._srv = rospy.Service(
            '~render', RenderVoxelGridArray, self._callback
        )

    def _callback(self, request):
        transform_msg = request.transform  # sensor to world
        cam_msg = request.camera_info
        depth_msg = request.depth
        grids_msg = request.grids

        if transform_msg.child_frame_id == cam_msg.header.frame_id:
            # cam2base
            quaternion, translation = objslampp.ros.from_ros_transform(
                transform_msg.transform
            )
            T_cam2base = objslampp.functions.transformation_matrix(
                quaternion, translation
            ).array
            T_base2cam = np.linalg.inv(T_cam2base)
        else:
            # base2cam
            quaternion, translation = objslampp.ros.from_ros_transform(
                transform_msg.transform
            )
            T_base2cam = objslampp.functions.transformation_matrix(
                quaternion, translation
            ).array

        bridge = cv_bridge.CvBridge()
        depth = bridge.imgmsg_to_cv2(depth_msg)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000
            depth[depth == 0] = np.nan
        assert depth.dtype == np.float32

        meshes = {}
        for grid in grids_msg.grids:
            mesh = grid_msg_to_mesh(grid)
            if mesh is None:
                continue
            mesh.apply_transform(T_base2cam)
            meshes[grid.instance_id] = mesh

        ins = np.full((cam_msg.height, cam_msg.width), -2, np.int32)
        if meshes:
            tmp_dir = path.Path(tempfile.mkdtemp())
            pybullet.connect(pybullet.DIRECT)

            uniq_id_to_ins_id = {}
            for instance_id, mesh in meshes.items():
                mesh_file = tmp_dir / f'{instance_id:04d}.obj'
                trimesh.exchange.export.export_mesh(mesh, mesh_file)

                uniq_id = objslampp.extra.pybullet.add_model(
                    visual_file=mesh_file,
                    collision_file=mesh_file,
                    register=False,
                )
                uniq_id_to_ins_id[uniq_id] = instance_id

            K = np.array(cam_msg.K).reshape(3, 3)
            camera = trimesh.scene.Camera(
                resolution=(cam_msg.width, cam_msg.height),
                focal=(K[0, 0], K[1, 1])
            )
            _, depth_rend, uniq = objslampp.extra.pybullet.render_camera(
                np.eye(4),
                fovy=camera.fov[1],
                height=camera.resolution[1],
                width=camera.resolution[0],
            )

            pybullet.disconnect()
            tmp_dir.rmtree_p()

            for uniq_id, ins_id in uniq_id_to_ins_id.items():
                ins[uniq == uniq_id] = ins_id
            with np.errstate(invalid='ignore'):
                ins[(uniq != -1) &
                    (np.isnan(depth) | (depth_rend > depth))] = -2

        bridge = cv_bridge.CvBridge()
        ins_msg = bridge.cv2_to_imgmsg(ins)
        ins_msg.header = cam_msg.header
        return RenderVoxelGridArrayResponse(ins_msg)


if __name__ == '__main__':
    rospy.init_node('render_voxel_grids')
    RenderVoxelGrids()
    rospy.spin()
