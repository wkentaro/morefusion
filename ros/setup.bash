OBJSLAMPP_PREFIX=$HOME/ros_objslampp/src/objslampp
ROSOBJSLAMPP_PREFIX=$HOME/ros_objslampp

# FIXME: need to load multiple times in some reasons
for i in seq 0 1; do

  unset CMAKE_PREFIX_PATH
  unset PYTHONPATH

  source $OBJSLAMPP_PREFIX/.anaconda3/bin/activate
  source /opt/ros/kinetic/setup.bash
  source $ROSOBJSLAMPP_PREFIX/devel/setup.bash

done
