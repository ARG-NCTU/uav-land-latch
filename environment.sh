#source /opt/ros/melodic/setup.bash
source $HOME/robotx-2022/catkin_ws/devel/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/robotx-2022/Firmware
. ~/robotx-2022/Firmware/Tools/setup_gazebo.bash ~/robotx-2022/Firmware ~/robotx-2022/Firmware/build/px4_sitl_default

source set_ros_master.sh 127.0.0.1
source set_ros_ip.sh 127.0.0.1
