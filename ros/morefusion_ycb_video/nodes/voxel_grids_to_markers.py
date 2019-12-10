#!/usr/bin/env python

import imgviz
import numpy as np
import trimesh

from geometry_msgs.msg import Point
from ros_objslampp_ycb_video.msg import VoxelGridArray
import rospy
import topic_tools
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray


class VoxelGridsToMarkers(topic_tools.LazyTransport):

    _colormap = imgviz.label_colormap()

    def __init__(self):
        super().__init__()
        self._show_bbox = rospy.get_param('~show_bbox', True)
        self._pub = self.advertise('~output', MarkerArray, queue_size=1)
        self._post_init()

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
        markers = MarkerArray()

        marker = Marker(
            header=grids_msg.header,
            id=-1,
            action=Marker.DELETEALL
        )
        markers.markers.append(marker)

        for grid in grids_msg.grids:
            instance_id = grid.instance_id
            color = self._colormap[instance_id + 1]

            origin = np.array([grid.origin.x, grid.origin.y, grid.origin.z])
            indices = np.array(grid.indices)
            k = indices % grid.dims.z
            j = indices // grid.dims.z % grid.dims.y
            i = indices // grid.dims.z // grid.dims.y
            indices = np.column_stack((i, j, k))
            points = origin + indices * grid.pitch
            dims = np.array([grid.dims.x, grid.dims.y, grid.dims.z])

            marker = Marker()
            marker.ns = f'{instance_id}'
            marker.id = (instance_id + 1) * 2
            marker.header = grids_msg.header
            marker.action = Marker.ADD
            marker.type = Marker.CUBE_LIST
            marker.points = [Point(*p) for p in points]
            marker.scale.x = grid.pitch
            marker.scale.y = grid.pitch
            marker.scale.z = grid.pitch
            marker.color.r = color[0] / 255.
            marker.color.g = color[1] / 255.
            marker.color.b = color[2] / 255.
            marker.color.a = 1
            markers.markers.append(marker)

            if not self._show_bbox:
                continue

            center = origin + (dims / 2 - 0.5) * grid.pitch
            extents = dims * grid.pitch
            bbox = trimesh.path.creation.box_outline(extents)
            bbox.apply_translation(center)

            marker = Marker()
            marker.ns = f'{instance_id}'
            marker.id = (instance_id + 1) * 2 + 1
            marker.header = grids_msg.header
            marker.action = Marker.ADD
            marker.type = Marker.LINE_LIST
            marker.points = [
                Point(*bbox.vertices[i])
                for node in bbox.vertex_nodes for i in node
            ]
            marker.color.r = color[0] / 255.
            marker.color.g = color[1] / 255.
            marker.color.b = color[2] / 255.
            marker.color.a = 1
            marker.scale.x = 0.005
            markers.markers.append(marker)
        self._pub.publish(markers)


if __name__ == '__main__':
    rospy.init_node('voxel_grids_to_markers')
    VoxelGridsToMarkers()
    rospy.spin()
