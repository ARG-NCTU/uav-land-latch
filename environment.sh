#source /opt/ros/melodic/setup.bash
source $HOME/uav-land-latch/catkin_ws/devel/setup.bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:~/uav-land-latch/Firmware
. ~/uav-land-latch/Firmware/Tools/setup_gazebo.bash ~/uav-land-latch/Firmware ~/uav-land-latch/Firmware/build/px4_sitl_default

source set_ros_master.sh 127.0.0.1
source set_ros_ip.sh 127.0.0.1
export REPO_NAME=uav-land-latch