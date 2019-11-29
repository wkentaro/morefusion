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

- `robot-node`: Computer with Real-time OS for Panda robot.
- `robot-agent`: Computer for visual processing and motion planning.

#### @robot-node

```bash
mkdir -p ~/ros_objslampp/src
cd ~/ros_objslampp/src

git clone https://github.com/wkentaro/objslampp.git

cd ~/ros_objslampp
ln -s src/ros/*.sh .

./catkin_build.robot_node.sh
source devel/setup.bash

rosrun franka_control_custom create_udev_rules.sh
```

#### @robot-agent

```bash
mkdir -p ~/ros_objslampp/src
cd ~/ros_objslampp/src

git clone https://github.com/wkentaro/objslampp.git
cd objslampp
make install

cd ~/ros_objslampp
ln -s src/ros/*.sh .

./rosdep_install.sh
./catkin_build.sh

ln -s src/objslampp/ros/autoenv.zsh .autoenv.zsh
ln -s src/objslampp/ros/autoenv_leave.zsh .autoenv_leave.zsh
source .autoenv.zsh
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

<div>
  <img src="https://drive.google.com/uc?id=1BbjWZPTZhoqbsH4OlzIghOO0VZhG69mK" width="60%" />
  <br/>
  <b><i>Figure 1. Static Scene Reconstruction with the Hand-mounted Camera.</i></b>
</div>

#### Robotic Pick-and-Place

```bash
robot-agent $ sudo ntpdata 0.uk.pool.ntp.org  # for time synchronization
robot-node  $ sudo ntpdata 0.uk.pool.ntp.org  # for time synchronization

robot-node  $ roscore

robot-agent $ roslaunch franka_moveit_custom objslampp_demo1.launch

robot-node  $ roslaunch ros_objslampp_ycb_video rs_rgbd.robot.launch
robot-node  $ roslaunch ros_objslampp_ycb_video rviz_static.launch
robot-node  $ roslaunch ros_objslampp_ycb_video setup_static.robot.launch TARGET:=2
robot-node  $ rosrun ros_objslampp_ycb_video robot_demo_node.py
>>> ri.run()
```

<div>
  <img src="https://drive.google.com/uc?id=1JeIlT2yyhruR5DreFbI9htP8N4X4fP10" width="30%" />
  <img src="https://drive.google.com/uc?id=1vO0k7NS0iRkzGhcmGHBpqe8sp7_i-n0a" width="30%" />
  <img src="https://drive.google.com/uc?id=1aj657Z8_T4JR4ceEh0laiP88ggBllYPK" width="30%" />
  <br/>
  <b><i>Figure 2. (a) Scanning the Scene; (b) Removing Distractor Objects; (c) Picking Target Object.</i></b>
</div>
