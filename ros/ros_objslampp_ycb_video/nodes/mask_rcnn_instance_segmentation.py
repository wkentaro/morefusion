#!/usr/bin/env python

from chainercv.links.model.fpn import MaskRCNNFPNResNet50
import gdown
import numpy as np

import objslampp

import cv_bridge
import rospy
from ros_objslampp_msgs.msg import ObjectClass
from ros_objslampp_msgs.msg import ObjectClassArray
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
            '~output/class', ObjectClassArray, queue_size=1
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
        del labels

        if self._context:
            keep = np.isin(class_ids, self._context)
            masks = masks[keep]
            class_ids = class_ids[keep]
            confs = confs[keep]

        if len(masks) > 0:
            keep = masks.sum(axis=(1, 2)) > 0
            class_ids = class_ids[keep]
            masks = masks[keep]
            confs = confs[keep]

        if len(confs) > 0:
            sort = np.argsort(confs)
            class_ids = class_ids[sort]
            masks = masks[sort]
            confs = confs[sort]

        instance_ids = np.arange(0, len(masks))
        label_ins = np.full(rgb.shape[:2], -1, dtype=np.int32)
        for ins_id, mask in zip(instance_ids, masks):
            label_ins[mask] = ins_id
        ins_msg = bridge.cv2_to_imgmsg(label_ins)
        ins_msg.header = imgmsg.header
        self._pub_ins.publish(ins_msg)

        instance_ids_active = np.unique(label_ins)
        keep = np.isin(instance_ids, instance_ids_active)
        instance_ids = instance_ids[keep]
        class_ids = class_ids[keep]
        confs = confs[keep]

        cls_msg = ObjectClassArray(
            header=imgmsg.header
        )
        for ins_id, cls_id, conf in zip(instance_ids, class_ids, confs):
            cls_msg.classes.append(ObjectClass(
                instance_id=ins_id,
                class_id=cls_id,
                confidence=conf,
            ))

        self._pub_cls.publish(cls_msg)


if __name__ == '__main__':
    rospy.init_node('mask_rcnn_instance_segmentation')
    MaskRCNNInstanceSegmentationNode()
    rospy.spin()
