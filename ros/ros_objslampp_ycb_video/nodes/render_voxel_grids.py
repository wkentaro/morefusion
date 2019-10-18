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
from topic_tools import LazyTransport
from ros_objslampp_msgs.msg import VoxelGridArray

from voxel_grids_to_mesh_markers import grid_msg_to_mesh


class RenderVoxelGrids(LazyTransport):

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', Image, queue_size=1)
        self._post_init()

    def subscribe(self):
        sub_cam = message_filters.Subscriber(
            '~input/camera_info', CameraInfo, queue_size=1
        )
        sub_grids = message_filters.Subscriber(
            '~input/grids', VoxelGridArray, queue_size=1
        )
        self._subscribers = [sub_cam, sub_grids]
        sync = message_filters.ApproximateTimeSynchronizer(
            self._subscribers, queue_size=100, slop=0.1, allow_headerless=True
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, cam_msg, grids_msg):
        pybullet.connect(pybullet.DIRECT)

        tmp_dir = path.Path(tempfile.mkdtemp())

        uniq_id_to_ins_id = {}
        for grid in grids_msg.grids:
            mesh = grid_msg_to_mesh(grid)
            if mesh is None:
                continue

            mesh_file = tmp_dir / f'{grid.instance_id:04d}.obj'
            trimesh.exchange.export.export_mesh(mesh, mesh_file)

            uniq_id = objslampp.extra.pybullet.add_model(
                visual_file=mesh_file,
                collision_file=mesh_file,
                register=False,
            )
            uniq_id_to_ins_id[uniq_id] = grid.instance_id

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

        ins = np.full_like(uniq, -1)
        for uniq_id, ins_id in uniq_id_to_ins_id.items():
            ins[uniq == uniq_id] = ins_id

        bridge = cv_bridge.CvBridge()
        ins_msg = bridge.cv2_to_imgmsg(ins)
        ins_msg.header = grids_msg.header
        self._pub.publish(ins_msg)


if __name__ == '__main__':
    rospy.init_node('render_voxel_grids')
    RenderVoxelGrids()
    rospy.spin()
