<h1 align="center">
  objslampp
</h1>

<h4 align="center">
  Volumetric fusion and CAD alignment for object-level SLAM
</h4>

<div align="center">
  <a href="https://github.com/wkentaro/objslampp/actions">
    <img src="https://github.com/wkentaro/objslampp/workflows/CI/badge.svg">
  </a>
</div>


## Installation

### Python Project

```bash
make install
```


### ROS Project

```bash
mkdir -p ~/ros_objslampp/src
cd ~/ros_objslampp/src

git clone https://github.com/wkentaro/objslampp.git
cd objslampp
make install

cd ~/ros_objslampp
ln -s src/objslampp/ros/catkin_init.sh

./catkin_init.sh

ln -s src/objslampp/ros/autoenv.zsh .autoenv.zsh
ln -s src/objslampp/ros/autoenv_leave.zsh .autoenv_leave.zsh
source .autoenv.zsh

catkin build ros_objslampp_ycb_video
```


### Demonstrations

#### Dynamic Scene

```bash
roslaunch ros_objslampp_ycb_video rs_rgbd.robot.launch
roslaunch ros_objslampp_ycb_video rviz_dynamic.robot.launch
roslaunch ros_objslampp_ycb_video setup_dynamic.robot.launch
```

#### Static Scene

```bash
roslaunch ros_objslampp_ycb_video rs_rgbd.robot.launch
roslaunch ros_objslampp_ycb_video rviz_static.robot.launch
roslaunch ros_objslampp_ycb_video setup_static.robot.launch
```

<img src="https://drive.google.com/uc?id=1BbjWZPTZhoqbsH4OlzIghOO0VZhG69mK" />
