#!/usr/bin/env python

import json

import chainer
import numpy as np
import path
import imgviz
import trimesh.transformations as tf

import objslampp
import objslampp.contrib.singleview_3d as contrib

import cv_bridge
import rospy
from topic_tools import LazyTransport
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker, MarkerArray

import sys
sys.path.insert(0, '/home/wkentaro/objslampp/examples/ycb_video/preliminary')  # NOQA
from preliminary import ICPRegistration


class SingleViewPoseEstimation3D(LazyTransport):

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(self):
        pretrained_model = '/home/wkentaro/objslampp/examples/ycb_video/singleview_3d/logs.20190930.all_data/20191001_093434.926883190/snapshot_model_best_add.npz'  # NOQA
        args_file = path.Path(pretrained_model).parent / 'args'

        with open(args_file) as f:
            args_data = json.load(f)

        self._model = contrib.models.Model(
            n_fg_class=len(args_data['class_names'][1:]),
            pretrained_resnet18=args_data['pretrained_resnet18'],
            with_occupancy=args_data['with_occupancy'],
            loss=args_data['loss'],
            loss_scale=args_data['loss_scale'],
        )
        chainer.serializers.load_npz(pretrained_model, self._model)
        self._model.to_gpu()

        super().__init__()
        self._pub_debug_rgbd = self.advertise(
            '~output/debug/rgbd', Image, queue_size=1
        )
        self._pub_markers = self.advertise(
            '~output/markers', MarkerArray, queue_size=1
        )

    def subscribe(self):
        sub_cam = message_filters.Subscriber('~input/camera_info', CameraInfo)
        sub_rgb = message_filters.Subscriber(
            '~input/rgb', Image, queue_size=1, buff_size=2 ** 24
        )
        sub_depth = message_filters.Subscriber(
            '~input/depth', Image, queue_size=1, buff_size=2 ** 24
        )
        sub_ins = message_filters.Subscriber(
            '~input/label_ins', Image, queue_size=1, buff_size=2 ** 24
        )
        sub_cls = message_filters.Subscriber(
            '~input/label_cls', Image, queue_size=1, buff_size=2 ** 24
        )
        self._subscribers = [sub_cam, sub_rgb, sub_depth, sub_ins, sub_cls]
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=200
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(self, cam_msg, rgb_msg, depth_msg, ins_msg, cls_msg):
        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
        depth = bridge.imgmsg_to_cv2(depth_msg)
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000
            depth[depth == 0] = np.nan
        assert depth.dtype == np.float32
        K = np.array(cam_msg.K).reshape(3, 3)
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        ins = bridge.imgmsg_to_cv2(ins_msg)
        cls = bridge.imgmsg_to_cv2(cls_msg)

        instance_ids = np.unique(ins)
        instance_ids = instance_ids[instance_ids >= 0]

        examples = []
        nanmask = np.isnan(pcd).any(axis=2)
        for ins_id in instance_ids:
            mask = ins == ins_id
            if (~nanmask & mask).sum() < 50:
                continue
            unique, counts = np.unique(cls[mask], return_counts=True)
            cls_id = unique[np.argmax(counts)]
            bbox = objslampp.geometry.masks_to_bboxes([mask])[0]
            y1, x1, y2, x2 = bbox.round().astype(int)
            rgb_ins = rgb[y1:y2, x1:x2].copy()
            rgb_ins[~mask[y1:y2, x1:x2]] = 0
            rgb_ins = imgviz.centerize(rgb_ins, (256, 256), cval=0)
            pcd_ins = pcd[y1:y2, x1:x2].copy()
            pcd_ins[~mask[y1:y2, x1:x2]] = np.nan
            pcd_ins = imgviz.centerize(
                pcd_ins, (256, 256), cval=np.nan, interpolation='nearest'
            )
            examples.append(dict(
                class_id=cls_id,
                rgb=rgb_ins,
                pcd=pcd_ins,
            ))
        if not examples:
            return
        inputs = chainer.dataset.concat_examples(examples, device=0)

        if self._pub_debug_rgbd.get_num_connections() > 0:
            debug_rgbd = [
                imgviz.tile(
                    [e['rgb'], imgviz.depth2rgb(e['pcd'][:, :, 2])],
                    (1, 2)
                )
                for e in examples
            ]
            debug_rgbd = imgviz.tile(debug_rgbd, border=(255, 255, 255))
            debug_rgbd_msg = bridge.cv2_to_imgmsg(debug_rgbd, encoding='rgb8')
            debug_rgbd_msg.header = rgb_msg.header
            self._pub_debug_rgbd.publish(debug_rgbd_msg)

        with chainer.no_backprop_mode(), chainer.using_config('train', False):
            quaternion, translation, confidence = self._model.predict(**inputs)
        indices = confidence.array.argmax(axis=1)
        B = quaternion.shape[0]
        quaternion = quaternion[np.arange(B), indices]
        translation = translation[np.arange(B), indices]
        quaternion = chainer.cuda.to_cpu(quaternion.array)
        translation = chainer.cuda.to_cpu(translation.array)

        transforms = objslampp.functions.transformation_matrix(
            quaternion, translation
        ).array
        for i in range(B):
            pcd_cad = self._models.get_pcd(examples[i]['class_id'])
            pcd_depth = examples[i]['pcd']
            pcd_depth = pcd_depth[~np.isnan(pcd_depth).any(axis=2)]
            icp = ICPRegistration(
                pcd_depth=pcd_depth,
                pcd_cad=pcd_cad,
                transform_init=transforms[i],
            )
            transform = icp.register()
            quaternion[i] = tf.quaternion_from_matrix(transform)
            translation[i] = tf.translation_from_matrix(transform)
        del transforms

        markers = MarkerArray()
        max_n_objects = 16
        for i in range(max_n_objects):
            marker = Marker()
            marker.header = rgb_msg.header
            marker.ns = '/singleview_3d_pose_estimation'
            marker.id = i
            if i >= B:
                marker.action = Marker.DELETE
            else:
                cls_id = examples[i]['class_id']
                marker.action = Marker.ADD
                marker.type = Marker.MESH_RESOURCE
                marker.pose.position.x = translation[i][0]
                marker.pose.position.y = translation[i][1]
                marker.pose.position.z = translation[i][2]
                marker.pose.orientation.x = quaternion[i][1]
                marker.pose.orientation.y = quaternion[i][2]
                marker.pose.orientation.z = quaternion[i][3]
                marker.pose.orientation.w = quaternion[i][0]
                marker.scale.x = 1
                marker.scale.y = 1
                marker.scale.z = 1
                cad_file = self._models.get_cad_file(cls_id)
                marker.mesh_resource = f'file://{cad_file}'
                marker.mesh_use_embedded_materials = True
            markers.markers.append(marker)
        self._pub_markers.publish(markers)


if __name__ == '__main__':
    rospy.init_node('singleview_3d_pose_estimation')
    SingleViewPoseEstimation3D()
    rospy.spin()
