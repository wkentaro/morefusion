// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_STL_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_STL_H_

#include <map>
#include <vector>

namespace ros_objslampp_ycb_video {
namespace utils {

template<typename T, typename U>
std::vector<T> keys(std::map<T, U> map) {
  std::vector<T> res;
  for (typename std::map<T, U>::iterator it = map.begin(); it != map.end(); it++) {
    res.push_back(it->first);
  }
  return res;
}

}  // namespace utils
}  // namespace ros_objslampp_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_STL_H_
