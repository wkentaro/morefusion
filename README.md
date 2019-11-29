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
ln -s *.sh

./catkin_init.sh
./install_realsense.sh
./rosdep_install.sh

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

#### Robotic Pick-and-Place

```bash
robot-agent $ sudo ntpdata 0.uk.pool.ntp.org  # for time synchronization
robot-node  $ sudo ntpdata 0.uk.pool.ntp.org  # for time synchronization

robot-node  $ roscore

robot-agent $ sudo chmod 777 /dev/ttyACM0  # for serial control of suction gripper
robot-agent $ roslaunch franka_moveit_custom objslampp_demo1.launch

robot-node  $ roslaunch ros_objslampp_ycb_video rs_rgbd.robot.launch
robot-node  $ roslaunch ros_objslampp_ycb_video rviz_static.launch
robot-node  $ roslaunch ros_objslampp_ycb_video setup_static.robot.launch TARGET:=2
robot-node  $ rosrun ros_objslampp_ycb_video robot_demo_node.py
>>> ri.run()
```

<img src="https://user-images.githubusercontent.com/4310419/69835651-1604b880-123b-11ea-93aa-2b65f7c284d0.jpg" width="30%" />
