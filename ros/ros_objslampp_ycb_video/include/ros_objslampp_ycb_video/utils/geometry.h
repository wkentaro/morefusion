// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

#include <vector>
#include <map>
#include <utility>

#include <opencv2/opencv.hpp>

#include "ros_objslampp_ycb_video/utils/opencv.h"


namespace ros_objslampp_ycb_video {
namespace utils {

void track_instance_id(
    const cv::Mat& reference,
    cv::Mat* target,
    std::map<int, unsigned>* instance_id_to_class_id) {
  std::vector<int> instance_ids1 = ros_objslampp_ycb_video::utils::unique<int>(reference);
  std::vector<int> instance_ids2 = ros_objslampp_ycb_video::utils::unique<int>(*target);

  std::map<int, std::pair<int, float> > ins_id2to1;
  for (size_t i = 0; i < instance_ids1.size(); i++) {
    // ins_id1: instance_id in the map
    int ins_id1 = instance_ids1[i];
    if (ins_id1 < 0) {
      continue;
    }

    cv::Mat mask1 = reference == ins_id1;
    for (size_t j = 0; j < instance_ids2.size(); j++) {
      // ins_id2: instance_id in the mask-rcnn output
      int ins_id2 = instance_ids2[j];
      if (ins_id2 < 0) {
        continue;
      }

      cv::Mat mask2 = (*target) == ins_id2;
      cv::Mat mask_intersection, mask_union;
      cv::bitwise_and(mask1, mask2, mask_intersection);
      cv::bitwise_or(mask1, mask2, mask_union);
      float iou = (cv::sum(mask_intersection) / cv::sum(mask_union))[0];
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
      if ((it2 == ins_id2to1.end())) {
        ins_id2to1.insert(std::make_pair(ins_id2, std::make_pair(ins_id1, iou)));
      } else if (iou > it2->second.second) {
        it2->second = std::make_pair(ins_id1, iou);
      }
    }
  }

  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id->begin();
       it != instance_id_to_class_id->end(); it++) {
    // ins_id2: instance_id in mask-rcnn output
    int ins_id2 = it->first;
    unsigned class_id = it->second;
    std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
    if (it2 != ins_id2to1.end()) {
      int ins_id1 = it2->second.first;
      if (ins_id1 != ins_id2) {
        instance_id_to_class_id->erase(it);
        instance_id_to_class_id->insert(std::make_pair(ins_id1, class_id));
      }
    }
  }

  for (size_t j = 0; j < target->rows; j++) {
    for (size_t i = 0; i < target->cols; i++) {
      int ins_id2 = target->at<int>(j, i);
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
      if (it2 != ins_id2to1.end()) {
        int ins_id1 = it2->second.first;
        if (ins_id1 != ins_id2) {
          target->at<int>(j, i) = ins_id1;
        }
      }
    }
  }
}

}  // namespace utils

}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
