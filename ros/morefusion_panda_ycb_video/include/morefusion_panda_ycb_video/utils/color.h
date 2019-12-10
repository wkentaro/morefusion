// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_COLOR_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_COLOR_H_

#include <std_msgs/ColorRGBA.h>

namespace morefusion_panda_ycb_video {
namespace utils {

std_msgs::ColorRGBA colorCategory40(int i) {
  std_msgs::ColorRGBA c;
  c.a = 1.0;
  switch (i % 40) {
  case 0:
  {
    c.r = 0.000000;
    c.g = 0.000000;
    c.b = 0.000000;
  }
  break;
  case 1:
  {
    c.r = 0.501961;
    c.g = 0.000000;
    c.b = 0.000000;
  }
  break;
  case 2:
  {
    c.r = 0.000000;
    c.g = 0.501961;
    c.b = 0.000000;
  }
  break;
  case 3:
  {
    c.r = 0.501961;
    c.g = 0.501961;
    c.b = 0.000000;
  }
  break;
  case 4:
  {
    c.r = 0.000000;
    c.g = 0.000000;
    c.b = 0.501961;
  }
  break;
  case 5:
  {
    c.r = 0.501961;
    c.g = 0.000000;
    c.b = 0.501961;
  }
  break;
  case 6:
  {
    c.r = 0.000000;
    c.g = 0.501961;
    c.b = 0.501961;
  }
  break;
  case 7:
  {
    c.r = 0.501961;
    c.g = 0.501961;
    c.b = 0.501961;
  }
  break;
  case 8:
  {
    c.r = 0.250980;
    c.g = 0.000000;
    c.b = 0.000000;
  }
  break;
  case 9:
  {
    c.r = 0.752941;
    c.g = 0.000000;
    c.b = 0.000000;
  }
  break;
  case 10:
  {
    c.r = 0.250980;
    c.g = 0.501961;
    c.b = 0.000000;
  }
  break;
  case 11:
  {
    c.r = 0.752941;
    c.g = 0.501961;
    c.b = 0.000000;
  }
  break;
  case 12:
  {
    c.r = 0.250980;
    c.g = 0.000000;
    c.b = 0.501961;
  }
  break;
  case 13:
  {
    c.r = 0.752941;
    c.g = 0.000000;
    c.b = 0.501961;
  }
  break;
  case 14:
  {
    c.r = 0.250980;
    c.g = 0.501961;
    c.b = 0.501961;
  }
  break;
  case 15:
  {
    c.r = 0.752941;
    c.g = 0.501961;
    c.b = 0.501961;
  }
  break;
  case 16:
  {
    c.r = 0.000000;
    c.g = 0.250980;
    c.b = 0.000000;
  }
  break;
  case 17:
  {
    c.r = 0.501961;
    c.g = 0.250980;
    c.b = 0.000000;
  }
  break;
  case 18:
  {
    c.r = 0.000000;
    c.g = 0.752941;
    c.b = 0.000000;
  }
  break;
  case 19:
  {
    c.r = 0.501961;
    c.g = 0.752941;
    c.b = 0.000000;
  }
  break;
  case 20:
  {
    c.r = 0.000000;
    c.g = 0.250980;
    c.b = 0.501961;
  }
  break;
  case 21:
  {
    c.r = 0.501961;
    c.g = 0.250980;
    c.b = 0.501961;
  }
  break;
  case 22:
  {
    c.r = 0.000000;
    c.g = 0.752941;
    c.b = 0.501961;
  }
  break;
  case 23:
  {
    c.r = 0.501961;
    c.g = 0.752941;
    c.b = 0.501961;
  }
  break;
  case 24:
  {
    c.r = 0.250980;
    c.g = 0.250980;
    c.b = 0.000000;
  }
  break;
  case 25:
  {
    c.r = 0.752941;
    c.g = 0.250980;
    c.b = 0.000000;
  }
  break;
  case 26:
  {
    c.r = 0.250980;
    c.g = 0.752941;
    c.b = 0.000000;
  }
  break;
  case 27:
  {
    c.r = 0.752941;
    c.g = 0.752941;
    c.b = 0.000000;
  }
  break;
  case 28:
  {
    c.r = 0.250980;
    c.g = 0.250980;
    c.b = 0.501961;
  }
  break;
  case 29:
  {
    c.r = 0.752941;
    c.g = 0.250980;
    c.b = 0.501961;
  }
  break;
  case 30:
  {
    c.r = 0.250980;
    c.g = 0.752941;
    c.b = 0.501961;
  }
  break;
  case 31:
  {
    c.r = 0.752941;
    c.g = 0.752941;
    c.b = 0.501961;
  }
  break;
  case 32:
  {
    c.r = 0.000000;
    c.g = 0.000000;
    c.b = 0.250980;
  }
  break;
  case 33:
  {
    c.r = 0.501961;
    c.g = 0.000000;
    c.b = 0.250980;
  }
  break;
  case 34:
  {
    c.r = 0.000000;
    c.g = 0.501961;
    c.b = 0.250980;
  }
  break;
  case 35:
  {
    c.r = 0.501961;
    c.g = 0.501961;
    c.b = 0.250980;
  }
  break;
  case 36:
  {
    c.r = 0.000000;
    c.g = 0.000000;
    c.b = 0.752941;
  }
  break;
  case 37:
  {
    c.r = 0.501961;
    c.g = 0.000000;
    c.b = 0.752941;
  }
  break;
  case 38:
  {
    c.r = 0.000000;
    c.g = 0.501961;
    c.b = 0.752941;
  }
  break;
  case 39:
  {
    c.r = 0.501961;
    c.g = 0.501961;
    c.b = 0.752941;
  }
  break;
  }
  return c;
}

}  // namespace utils
}  // namespace morefusion_panda_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_COLOR_H_
