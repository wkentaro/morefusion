#!/usr/bin/env python

import json

import chainer
import gdown
import numpy as np
import imgviz
import trimesh.transformations as ttf

import objslampp
import objslampp.contrib.singleview_3d as contrib

import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from ros_objslampp_msgs.msg import VoxelGridArray
from ros_objslampp_msgs.msg import ObjectPose
from ros_objslampp_msgs.msg import ObjectPoseArray
import topic_tools


class SingleViewPoseEstimation3D(topic_tools.LazyTransport):

    _models = objslampp.datasets.YCBVideoModels()

    def __init__(self):
        self._with_occupancy = rospy.get_param('~with_occupancy')
        if self._with_occupancy:
            pretrained_model = gdown.cached_download(
                url='https://drive.google.com/uc?id=1gNfkP7vY2LwnuaoV55He0fKdVdLFD5iR',  # NOQA
                md5='e516bd08791d892bab5374e575f82de4',
            )
            args_file = gdown.cached_download(
                url='https://drive.google.com/uc?id=1RcW0jGzmr3jp5SuEKaopd8aR9vknoqI3',  # NOQA
                md5='a495652fc2d9c1c951076c6e40e22815',
            )
        else:
            pretrained_model = gdown.cached_download(
                url='https://drive.google.com/uc?id=1Dv03xveUV3p3oFvlx1zwX6pWK56y_b-K',  # NOQA
                md5='94a988d4b9af9647f9e94a249212a40c',
            )
            args_file = gdown.cached_download(
                url='https://drive.google.com/uc?id=1z3CSQoYeUfg4KOUgtce4hAKkPuyZ161r',  # NOQA
                md5='67472da00c9687671b8e1af43b397071',
            )

        with open(args_file) as f:
            args_data = json.load(f)

        self._model = contrib.models.Model(
            n_fg_class=len(args_data['class_names'][1:]),
            pretrained_resnet18=args_data['pretrained_resnet18'],
            with_occupancy=args_data['with_occupancy'],
            # loss=args_data['loss'],
            # loss_scale=args_data['loss_scale'],
        )
        chainer.serializers.load_npz(pretrained_model, self._model)
        self._model.to_gpu()

        self._static = rospy.get_param('~static', False)
        self._rgb = None
        self._depth = None
        self._ins = None

        super().__init__()
        self._pub_debug_rgbd = self.advertise(
            '~output/debug/rgbd', Image, queue_size=1
        )
        self._pub_poses = self.advertise(
            '~output', ObjectPoseArray, queue_size=1
        )
        self._post_init()

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
            '~input/class', ClassificationResult, queue_size=1,
        )
        self._subscribers = [
            sub_cam,
            sub_rgb,
            sub_depth,
            sub_ins,
            sub_cls,
        ]
        if self._with_occupancy:
            sub_noentry = message_filters.Subscriber(
                '~input/grids_noentry', VoxelGridArray, queue_size=1,
            )
            self._subscribers.append(sub_noentry)
        sync = message_filters.TimeSynchronizer(
            self._subscribers, queue_size=100
        )
        sync.registerCallback(self._callback)

    def unsubscribe(self):
        for sub in self._subscribers:
            sub.unregister()

    def _callback(
        self, cam_msg, rgb_msg, depth_msg, ins_msg, cls_msg, noentry_msg=None
    ):
        bridge = cv_bridge.CvBridge()
        if self._rgb is None:
            rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='rgb8')
            if self._static:
                self._rgb = rgb
        else:
            rgb = self._rgb
        if self._depth is None:
            depth = bridge.imgmsg_to_cv2(depth_msg)
            if self._static:
                self._depth = depth
        else:
            depth = self._depth
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000
            depth[depth == 0] = np.nan
        assert depth.dtype == np.float32
        K = np.array(cam_msg.K).reshape(3, 3)
        pcd = objslampp.geometry.pointcloud_from_depth(
            depth, K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        )
        if self._ins is None:
            ins = bridge.imgmsg_to_cv2(ins_msg)
            if self._static:
                self._ins = ins
        else:
            ins = self._ins

        grids_noentry = {}
        if noentry_msg:
            for grid in noentry_msg.grids:
                instance_id = grid.instance_id
                dims = (grid.dims.x, grid.dims.y, grid.dims.z)
                indices = np.array(grid.indices)
                k = indices % grid.dims.z
                j = indices // grid.dims.z % grid.dims.y
                i = indices // grid.dims.z // grid.dims.y
                grid_nontarget_empty = np.zeros(dims, dtype=bool)
                grid_nontarget_empty[i, j, k] = True
                origin = np.array(
                    [grid.origin.x, grid.origin.y, grid.origin.z],
                    dtype=np.float32,
                )
                grids_noentry[instance_id] = dict(
                    origin=origin,
                    pitch=grid.pitch,
                    matrix=grid_nontarget_empty,
                )

        class_ids = cls_msg.labels
        instance_ids = np.arange(0, len(class_ids))

        examples = []
        keep = []
        nanmask = np.isnan(pcd).any(axis=2)
        for i, (ins_id, cls_id) in enumerate(zip(instance_ids, class_ids)):
            mask = ins == ins_id
            if (~nanmask & mask).sum() < 50:
                continue
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

            example = dict(
                class_id=cls_id,
                rgb=rgb_ins,
                pcd=pcd_ins,
            )
            if grids_noentry:
                example['origin'] = grids_noentry[ins_id]['origin']
                example['pitch'] = grids_noentry[ins_id]['pitch']
                example['grid_nontarget_empty'] = \
                    grids_noentry[ins_id]['matrix']
            examples.append(example)
            keep.append(i)
        if not examples:
            return
        inputs = chainer.dataset.concat_examples(examples, device=0)
        instance_ids = instance_ids[keep]
        del class_ids

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
            icp = objslampp.contrib.ICPRegistration(
                pcd_depth=pcd_depth,
                pcd_cad=pcd_cad,
                transform_init=transforms[i],
            )
            transform = icp.register()
            quaternion[i] = ttf.quaternion_from_matrix(transform)
            translation[i] = ttf.translation_from_matrix(transform)
        del transforms

        poses = ObjectPoseArray()
        poses.header = rgb_msg.header
        for i, (ins_id, example) in enumerate(zip(instance_ids, examples)):
            pose = ObjectPose()
            pose.pose.position.x = translation[i][0]
            pose.pose.position.y = translation[i][1]
            pose.pose.position.z = translation[i][2]
            pose.pose.orientation.w = quaternion[i][0]
            pose.pose.orientation.x = quaternion[i][1]
            pose.pose.orientation.y = quaternion[i][2]
            pose.pose.orientation.z = quaternion[i][3]
            pose.instance_id = ins_id
            pose.class_id = examples[i]['class_id']
            poses.poses.append(pose)
        self._pub_poses.publish(poses)


if __name__ == '__main__':
    rospy.init_node('singleview_3d_pose_estimation')
    SingleViewPoseEstimation3D()
    rospy.spin()
