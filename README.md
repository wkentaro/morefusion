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
ln -s src/ros/*.sh .

./rosdep_install.sh
./catkin_build.sh

ln -s src/objslampp/ros/autoenv.zsh .autoenv.zsh
ln -s src/objslampp/ros/autoenv_leave.zsh .autoenv_leave.zsh
source .autoenv.zsh
```

### ROS Project for Robotic Demonstrations

- `robot-agent`: A computer for visual processing and motion planning.
- `robot-node`: A computer with real-time OS for Panda robot.

#### @robot-agent

See <a href="#ros-project">above instruction</a>.

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

## Usage

### Dynamic Scene

```bash
roslaunch ros_objslampp_ycb_video rs_rgbd.robot.launch
roslaunch ros_objslampp_ycb_video rviz_dynamic.robot.launch
roslaunch ros_objslampp_ycb_video setup_dynamic.robot.launch
```

<div>
  <img src="https://drive.google.com/uc?id=1E3aqKf9TdSWDjL8rsbAe_oq_jEOQ5RbE" width="40%" />
  <br/>
  <i>Figure 1. Dynamic Scene Reconstruction with the Human Hand-mounted Camera.</i>
</div>

### Static Scene

```bash
# using orb-slam2 for camera tracking
roslaunch ros_objslampp_ycb_video rs_rgbd.launch
roslaunch ros_objslampp_ycb_video rviz_static.desk.launch
roslaunch ros_objslampp_ycb_video setup_static.desk.launch
```

<div>
  <img src="https://drive.google.com/uc?id=1s9gQguthVAQTacO6PaGQw4kOQrdlucri" width="40%" />
  <br/>
  <i>Figure 2. Static Scene Reconstruction with the Human Hand-mounted Camera.</i>
</div>

```bash
# using robotic kinematics for camera tracking
roslaunch ros_objslampp_ycb_video rs_rgbd.robot.launch
roslaunch ros_objslampp_ycb_video rviz_static.robot.launch
roslaunch ros_objslampp_ycb_video setup_static.robot.launch
```

<div>
  <img src="https://drive.google.com/uc?id=1BbjWZPTZhoqbsH4OlzIghOO0VZhG69mK" width="40%" />
  <br/>
  <i>Figure 3. Static Scene Reconstruction with the Robotic Hand-mounted Camera.</i>
</div>

### Robotic Pick-and-Place

```bash
robot-agent $ sudo ntpdata 0.uk.pool.ntp.org  # for time synchronization
robot-node  $ sudo ntpdata 0.uk.pool.ntp.org  # for time synchronization

robot-node  $ roscore

robot-agent $ roslaunch ros_objslampp_panda panda.launch

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
  <i>Figure 4. Targetted Object Pick-and-Place. (a) Scanning the Scene; (b) Removing Distractor Objects; (c) Picking Target Object.</i>
</div>
