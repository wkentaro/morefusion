#!/usr/bin/env python

import path
import tempfile

import numpy as np
import pybullet
import trimesh

import objslampp

import cv_bridge
import message_filters
import rospy
from sensor_msgs.msg import Image, CameraInfo
from ros_objslampp_msgs.msg import VoxelGridArray
import tf
import topic_tools

from voxel_grids_to_mesh_markers import grid_msg_to_mesh


class RenderVoxelGrids(topic_tools.LazyTransport):

    def __init__(self):
        super().__init__()
        self._base_frame = rospy.get_param('~frame_id', 'map')
        self._tf_listener = tf.TransformListener()
        self._pub = self.advertise('~output', Image, queue_size=1)
        self._post_init()

    def subscribe(self):
        sub_cam = message_filters.Subscriber(
            '~input/camera_info', CameraInfo, queue_size=1
        )
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1
        )
        sub_grids = message_filters.Subscriber(
            '~input/grids', VoxelGridArray, queue_size=1
        )
        self._subscribers = [sub_cam, sub_depth, sub_grids]
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=100
        )
        sync.registerCallback(self._callback)

    def _get_transform(self, sensor_frame, sensor_stamp):
        try:
            self._tf_listener.waitForTransform(
                self._base_frame,
                sensor_frame,
                sensor_stamp,
                timeout=rospy.Duration(1),
            )
        except Exception as e:
            rospy.logerr(e)
            return

        translation, quaternion = self._tf_listener.lookupTransform(
            self._base_frame,
            sensor_frame,
            sensor_stamp,
        )
        translation = np.asarray(translation)
        quaternion = np.asarray(quaternion)[[3, 0, 1, 2]]
        T_cam2base = objslampp.functions.transformation_matrix(
            quaternion, translation
        ).array
        return T_cam2base

    def _callback(self, cam_msg, depth_msg, grids_msg):
        assert grids_msg.header.frame_id == self._base_frame
        T_cam2base = self._get_transform(
            cam_msg.header.frame_id, cam_msg.header.stamp)
        if T_cam2base is None:
            return
        T_base2cam = np.linalg.inv(T_cam2base)

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
            ins[(uniq != -1) & (depth_rend > depth)] = -2

        bridge = cv_bridge.CvBridge()
        ins_msg = bridge.cv2_to_imgmsg(ins)
        ins_msg.header = cam_msg.header
        self._pub.publish(ins_msg)


if __name__ == '__main__':
    rospy.init_node('render_voxel_grids')
    RenderVoxelGrids()
    rospy.spin()
