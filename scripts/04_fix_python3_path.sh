#! /usr/bin/env bash
cp /home/argrobotx/uav-land-latch/catkin_ws/src/vrx/vrx_gazebo/nodes/ros_tools/fix_python3_path.py /usr/lib/python3.6

# This script is used to fix the python3 path when using procman.
# Procman add `/usr/local/lib/python2.7/dist-packages` at the beginning of the python3 path 
# but some python3 nodes need to use python3 packages
