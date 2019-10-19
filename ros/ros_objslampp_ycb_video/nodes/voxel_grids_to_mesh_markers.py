#!/usr/bin/env python

import tempfile

import imgviz
import numpy as np
import path
import scipy.ndimage
import trimesh

from visualization_msgs.msg import Marker, MarkerArray
from ros_objslampp_msgs.msg import VoxelGridArray
import rospy
import topic_tools


class VoxelGridsToMeshMarkers(topic_tools.LazyTransport):

    _colormap = imgviz.label_colormap()

    def __init__(self):
        super().__init__()
        self._pub = self.advertise('~output', MarkerArray, queue_size=1)
        self._tmp_dir = path.Path(tempfile.mkdtemp())
        self._post_init()

    def __del__(self):
        self._tmp_dir.rmtree_p()

    def subscribe(self):
        self._sub = rospy.Subscriber(
            '~input',
            VoxelGridArray,
            self._callback,
            queue_size=1,
            buff_size=2 ** 24,
        )

    def unsubscribe(self):
        self._sub.unregister()

    def _callback(self, grids_msg):
        markers_msg = MarkerArray()

        # marker = Marker(
        #     header=grids_msg.header,
        #     id=-1,
        #     action=Marker.DELETEALL
        # )
        # markers_msg.markers.append(marker)

        nsec = grids_msg.header.stamp.to_nsec()
        for grid in grids_msg.grids:
            mesh = grid_msg_to_mesh(grid)
            if mesh is None:
                continue

            mesh_file = self._tmp_dir / f'{nsec}_{grid.instance_id:04d}.ply'
            trimesh.exchange.export.export_mesh(mesh, mesh_file)

            color = self._colormap[grid.instance_id + 1]

            marker = Marker()
            marker.header = grids_msg.header
            marker.id = grid.instance_id
            marker.pose.orientation.w = 1
            marker.scale.x = 1
            marker.scale.y = 1
            marker.scale.z = 1
            marker.color.a = 1
            marker.color.r = color[0] / 255
            marker.color.g = color[1] / 255
            marker.color.b = color[2] / 255
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = f'file://{mesh_file}'
            marker.action = Marker.ADD
            markers_msg.markers.append(marker)
        self._pub.publish(markers_msg)


def grid_msg_to_mesh(grid):
    if not grid.indices:
        return

    pitch = grid.pitch
    origin = (grid.origin.x, grid.origin.y, grid.origin.z)
    dims = (grid.dims.x, grid.dims.y, grid.dims.z)

    matrix = np.zeros(dims, dtype=np.float32)
    matrix = matrix.flatten()
    matrix[list(grid.indices)] = grid.values
    matrix = matrix.reshape(dims)
    matrix = scipy.ndimage.morphology.grey_opening(
        matrix, size=(1, 1, 1)
    )
    matrix = scipy.ndimage.morphology.grey_dilation(
        matrix, size=(2, 2, 2)
    )
    mesh = trimesh.voxel.matrix_to_marching_cubes(
        matrix, pitch, origin
    )
    mesh = trimesh.smoothing.filter_humphrey(mesh)
    return mesh


if __name__ == '__main__':
    rospy.init_node('voxel_grids_to_mesh_markers')
    VoxelGridsToMeshMarkers()
    rospy.spin()
