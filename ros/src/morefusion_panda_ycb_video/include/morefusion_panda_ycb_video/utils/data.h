// Copyright (c) 2019 Kentaro Wada

#ifndef ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_DATA_H_
#define ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_DATA_H_

namespace morefusion_panda_ycb_video {
namespace utils {

double class_id_to_voxel_pitch(unsigned class_id) {
  double pitch;
  switch (class_id) {
  case  1: pitch = 0.006296589104319322; break;
  case  2: pitch = 0.008705823111730123; break;
  case  3: pitch = 0.006425726070431774; break;
  case  4: pitch = 0.004375644727606043; break;
  case  5: pitch = 0.007023497839423789; break;
  case  6: pitch = 0.003923674166124662; break;
  case  7: pitch = 0.006018916012848706; break;
  case  8: pitch = 0.004320481778555272; break;
  case  9: pitch = 0.004535342826373148; break;
  case 10: pitch = 0.006631487204390293; break;
  case 11: pitch = 0.009982031658204186; break;
  case 12: pitch = 0.008721623259758258; break;
  case 13: pitch = 0.007331656585392745; break;
  case 14: pitch = 0.005318687227615036; break;
  case 15: pitch = 0.008406278399464109; break;
  case 16: pitch = 0.0079006960844688; break;
  case 17: pitch = 0.00699458097945295; break;
  case 18: pitch = 0.0038783371057780278; break;
  case 19: pitch = 0.006648125743278138; break;
  case 20: pitch = 0.008405508709996566; break;
  case 21: pitch = 0.0033429720217908734; break;
  default:
  break;
  }
  return pitch;
}

}  // namespace utils
}  // namespace morefusion_panda_ycb_video

#endif  // ROS_ROS_OBJSLAMPP_YCB_VIDEO_INCLUDE_ROS_OBJSLAMPP_YCB_VIDEO_UTILS_DATA_H_
