// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_

#include <algorithm>
#include <vector>
#include <map>
#include <tuple>
#include <utility>

#include <opencv2/opencv.hpp>

#include "ros_objslampp_ycb_video/utils/opencv.h"


namespace ros_objslampp_ycb_video {
namespace utils {

void track_instance_id(
    const cv::Mat& reference,
    cv::Mat* target,
    std::map<int, unsigned>* instance_id_to_class_id,
    unsigned* instance_counter) {
  std::vector<int> instance_ids1 = ros_objslampp_ycb_video::utils::unique<int>(reference);
  std::vector<int> instance_ids2 = ros_objslampp_ycb_video::utils::unique<int>(*target);

  cv::Mat mask_nonedge = cv::Mat::zeros(reference.rows, reference.cols, CV_8UC1);
  cv::rectangle(
    mask_nonedge,
    cv::Point(reference.cols * 0.15, reference.rows * 0.15),
    cv::Point(reference.cols * 0.85, reference.rows * 0.85),
    /*color=*/255,
    /*thickness=*/CV_FILLED);
  cv::Mat mask_edge;
  cv::bitwise_not(mask_nonedge, mask_edge);

  // Compute IOU
  std::map<int, std::pair<int, float> > ins_id2to1;
  std::set<int> ins_ids2_on_edge;
  for (size_t i = 0; i < instance_ids2.size(); i++) {
    // ins_id2: instance_id in the mask-rcnn output
    int ins_id2 = instance_ids2[i];
    if (ins_id2 < 0) {
      continue;
    }

    cv::Mat mask2 = (*target) == ins_id2;
    ins_id2to1.insert(std::make_pair(ins_id2, std::make_pair(-1, 0)));

    cv::Mat mask_intersect_edge, mask_intersect_nonedge;
    cv::bitwise_and(mask_edge, mask2, mask_intersect_edge);
    cv::bitwise_and(mask_nonedge, mask2, mask_intersect_nonedge);
    if (cv::countNonZero(mask_intersect_edge) > cv::countNonZero(mask_intersect_nonedge)) {
      ins_ids2_on_edge.insert(ins_id2);
    }

    for (size_t j = 0; j < instance_ids1.size(); j++) {
      // ins_id1: instance_id in the map
      int ins_id1 = instance_ids1[j];
      if (ins_id1 < 0) {
        continue;
      }

      cv::Mat mask1 = reference == ins_id1;
      cv::Mat mask_intersection, mask_union;
      cv::bitwise_and(mask1, mask2, mask_intersection);
      cv::bitwise_or(mask1, mask2, mask_union);
      float iou =
        static_cast<float>(cv::countNonZero(mask_intersection)) /
        static_cast<float>(cv::countNonZero(mask_union));
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
      if (iou > it2->second.second) {
        it2->second = std::make_pair(ins_id1, iou);
      }
    }
  }

  // Initialize new object
  for (std::map<int, std::pair<int, float> >::iterator it = ins_id2to1.begin();
       it != ins_id2to1.end(); it++) {
    int ins_id2 = it->first;
    if (ins_ids2_on_edge.find(ins_id2) != ins_ids2_on_edge.end()) {
      // it's on the edge, so don't initialize
      continue;
    }
    int ins_id1 = it->second.first;
    float iou = it->second.second;
    if (iou < 0.5) {
      it->second.first = *instance_counter;
      // new instance
      (*instance_counter)++;
    }
  }

  std::map<int, unsigned> instance_id_to_class_id_updated;
  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id->begin();
       it != instance_id_to_class_id->end(); it++) {
    int ins_id2 = it->first;
    if (ins_ids2_on_edge.find(ins_id2) != ins_ids2_on_edge.end()) {
      // it's on the edge, so skip inserting instance_id_to_class_id_updated
      continue;
    }
    int ins_id1 = ins_id2to1.find(ins_id2)->second.first;
    unsigned class_id = it->second;
    instance_id_to_class_id_updated.insert(std::make_pair(ins_id1, class_id));
  }
  instance_id_to_class_id->clear();
  for (std::map<int, unsigned>::iterator it = instance_id_to_class_id_updated.begin();
       it != instance_id_to_class_id_updated.end(); it++) {
    instance_id_to_class_id->insert(std::make_pair(it->first, it->second));
  }

  for (size_t j = 0; j < target->rows; j++) {
    for (size_t i = 0; i < target->cols; i++) {
      int ins_id2 = target->at<int>(j, i);
      if (ins_id2 < 0) {
        if (mask_edge.at<uint8_t>(j, i) != 0) {
          target->at<int>(j, i) = -2;
        }
        continue;
      }
      if (ins_ids2_on_edge.find(ins_id2) != ins_ids2_on_edge.end()) {
        target->at<int>(j, i) = -2;
        continue;
      }
      std::map<int, std::pair<int, float> >::iterator it2 = ins_id2to1.find(ins_id2);
      assert(it2 != ins_id2to1.end());
      int ins_id1 = it2->second.first;
      if (ins_id1 != ins_id2) {
        target->at<int>(j, i) = ins_id1;
      }
    }
  }

  std::vector<int> instance_ids_active = ros_objslampp_ycb_video::utils::unique<int>(*target);
  for (size_t i = 0; i < instance_ids_active.size(); i++) {
    int ins_id2 = instance_ids_active[i];
    if (ins_id2 < 0) {
      continue;
    }
    cv::Mat mask = (*target) == ins_id2;
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    for (size_t j = 0; j < contours.size(); j++) {
      cv::drawContours(*target, contours, j, /*color=*/-2, /*thickness=*/10);
    }
  }
}

}  // namespace utils

}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_GEOMETRY_H_
