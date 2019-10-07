OBJSLAMPP_PREFIX=$HOME/ros_objslampp/src/objslampp
ROSOBJSLAMPP_PREFIX=$HOME/ros_objslampp

unset CMAKE_PREFIX_PATH
unset PYTHONPATH

source /opt/ros/kinetic/setup.bash
source $ROSOBJSLAMPP_PREFIX/devel/setup.bash
source $OBJSLAMPP_PREFIX/.anaconda3/bin/activate
