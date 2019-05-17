<h1 align="center">
  objslampp
</h1>

<h4 align="center">
  Volumetric fusion and CAD alignment for object-level SLAM
</h4>

<div align="center">
  <a href="https://travis-ci.com/wkentaro/objslampp">
    <img src="https://travis-ci.com/wkentaro/objslampp.svg?token=zM5rExyvuRoJThsnqHAF&branch=master">
  </a>
</div>


## Installation

### Python Project

```bash
make install
```


### ROS Project

```bash
OBJSLAMPP_PREFIX=~/objslampp

cd ~
mkdir -p ros_objslampp_py2 && cd ros_objslampp_py2
mkdir src
catkin init
cd src
ln -s ~/objslampp/ros/ros_objslampp_py2

cd ~
mkdir -p ros_objslampp && cd ros_objslampp
mkdir src
catkin init
catkin config -DPYTHON_EXECUTABLE=$OBJSLAMPP_PREFIX/.anaconda3/bin/python \
              -DPYTHON_INCLUDE_DIR=$OBJSLAMPP_PREFIX/.anaconda3/include/python3.7m \
              -DPYTHON_LIBRARY=$OBJSLAMPP_PREFIX/.anaconda3/lib/libpython3.7m.so
ln -s ~/objslampp/ros/ros_objslampp_ycb_video

cd ~/ros_objslampp
catkin build
ln -s ~/objslampp/.anaconda3/lib/python3.7/site-packages/cv2 devel/lib/python3/dist-packages/cv2
```


## Contents

- [System Overview](https://drive.google.com/open?id=1EnOtEawvWUcihlsnSrIbNeB5oE-UJGDv)
- [Dataset Synthesizing](https://github.com/wkentaro/scenenetrgb-d/tree/master/python/examples#object-level-slam-full-recon-and-physical-sim-for-navi-and-manip-in-heavy-clutter)
