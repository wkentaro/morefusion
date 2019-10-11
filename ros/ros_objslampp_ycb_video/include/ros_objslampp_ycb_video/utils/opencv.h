// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_OPENCV_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_OPENCV_H_

#include <vector>

#include <opencv2/opencv.hpp>

namespace ros_objslampp_ycb_video {
namespace utils {

template<typename T>
std::vector<T> unique(const cv::Mat& input) {
  std::vector<T> out;
  for (size_t j = 0; j < input.rows; ++j) {
    for (size_t i = 0; i < input.cols; ++i) {
      T value = input.at<T>(j, i);
      if (std::find(out.begin(), out.end(), value) == out.end()) {
        out.push_back(value);
      }
    }
  }
  return out;
}

}  // namespace utils
}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_OPENCV_H_
