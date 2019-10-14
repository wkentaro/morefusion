#!/usr/bin/env python

from chainercv.links.model.fpn import MaskRCNNFPNResNet50
import cv2
import gdown
import numpy as np

import objslampp

import cv_bridge
from jsk_recognition_msgs.msg import ClassificationResult
import rospy
from sensor_msgs.msg import Image
from topic_tools import LazyTransport


class MaskRCNNInstanceSegmentationNode(LazyTransport):

    def __init__(self):
        super().__init__()

        self._class_names = objslampp.datasets.ycb_video.class_names
        self._context = rospy.get_param('~context')

        pretrained_model = gdown.cached_download(
            url='https://drive.google.com/uc?id=1Ge2S9JudxC5ODdsrjOy5XoW7l7Zcz65E',  # NOQA
            md5='fc06b1292a7e99f9c1deb063accbf7ea',
        )
        self._model = MaskRCNNFPNResNet50(
            n_fg_class=len(self._class_names[1:]),
            pretrained_model=pretrained_model,
        )
        self._model.to_gpu()

        self._pub_cls = self.advertise(
            '~output/class', ClassificationResult, queue_size=1
        )
        self._pub_ins = self.advertise(
            '~output/label_ins', Image, queue_size=1
        )
        self._post_init()  # FIXME

    def subscribe(self):
        self._sub = rospy.Subscriber('~input', Image, callback=self.callback,
                                     queue_size=1, buff_size=2 ** 24)

    def unsubscribe(self):
        self._sub.unregister()

    def callback(self, imgmsg):
        bridge = cv_bridge.CvBridge()
        rgb = bridge.imgmsg_to_cv2(imgmsg, desired_encoding='rgb8')

        with objslampp.utils.timer():
            masks, labels, confs = self._model.predict(
                [rgb.astype(np.float32).transpose(2, 0, 1)]
            )
        masks = masks[0]
        labels = labels[0]
        confs = confs[0]

        class_ids = labels + 1
        class_names = objslampp.datasets.ycb_video.class_names
        del labels

        if self._context:
            keep = np.isin(class_ids, self._context)
            masks = masks[keep]
            class_ids = class_ids[keep]
            confs = confs[keep]

        mask_contours = []
        for mask in masks:
            contours, _ = cv2.findContours(
                mask.astype(np.uint8),
                cv2.RETR_TREE,
                method=cv2.CHAIN_APPROX_SIMPLE,
            )
            mask_contour = np.zeros(mask.shape, dtype=np.uint8)
            if contours:
                cv2.drawContours(
                    mask_contour,
                    contours,
                    contourIdx=-1,
                    color=1,
                    thickness=10,
                )
            mask_contours.append(mask_contour.astype(bool))
        mask_contours = np.array(mask_contours)

        masks = masks & ~mask_contours

        keep = masks.sum(axis=(1, 2)) > 0
        masks = masks[keep]
        class_ids = class_ids[keep]
        confs = confs[keep]

        label_ins = np.full(rgb.shape[:2], -1, dtype=np.int32)
        sort = np.argsort(confs)
        class_ids = class_ids[sort]
        masks = masks[sort]
        for ins_id, (cls_id, mask, mask_contour) in \
                enumerate(zip(class_ids, masks, mask_contours)):
            label_ins[mask] = ins_id
            label_ins[mask_contour] = -2

        ins_msg = bridge.cv2_to_imgmsg(label_ins)
        ins_msg.header = imgmsg.header
        self._pub_ins.publish(ins_msg)

        cls_msg = ClassificationResult()
        cls_msg.header = imgmsg.header
        cls_msg.labels = class_ids.tolist()
        cls_msg.label_names = [class_names[c] for c in class_ids]
        cls_msg.label_proba = confs.tolist()
        cls_msg.target_names = class_names.tolist()
        self._pub_cls.publish(cls_msg)


if __name__ == '__main__':
    rospy.init_node('mask_rcnn_instance_segmentation')
    MaskRCNNInstanceSegmentationNode()
    rospy.spin()
