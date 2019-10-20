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

ln -s src/objslampp/ros/template.autoenv.zsh .autoenv.zsh
ln -s src/objslampp/ros/template.autoenv_leave.zsh .autoenv_leave.zsh
source .autoenv.zsh

catkin build ros_objslampp_ycb_video
```


## Contents

- [System Overview](https://drive.google.com/open?id=1EnOtEawvWUcihlsnSrIbNeB5oE-UJGDv)
- [Dataset Synthesizing](https://github.com/wkentaro/scenenetrgb-d/tree/master/python/examples#object-level-slam-full-recon-and-physical-sim-for-navi-and-manip-in-heavy-clutter)
