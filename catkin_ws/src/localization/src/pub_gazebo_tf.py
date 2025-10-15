#! /usr/bin/env python
import rospy
import numpy as np
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import tf
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Vector3
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, GetModelState, GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest


class WheelTF(object):
    def __init__(self):
        self.broadcaster = tf.TransformBroadcaster()
        self.veh = rospy.get_param("veh")
        self.frame = self.veh + "/base_link"
        self.fram_name = rospy.get_param("~frame_name", self.frame)
        self.paraent_name = rospy.get_param("~parent_name", "map")
        self.pub_pose = rospy.Publisher("/gazebo/" + self.veh + "/pose", PoseStamped, queue_size=1)
        self.pub_odometry = rospy.Publisher('truth_map_odometry', Odometry, queue_size=1)
        self.get_model = rospy.ServiceProxy('/gazebo/get_model_state', GetModelState)

    def gazebo_odom(self):
        agent = ModelState()
        rospy.wait_for_service('/gazebo/get_model_state')
        try:
            agent = self.get_model(self.veh, '')
        except (rospy.ServiceException) as e:
            print(e)
        new_pos = np.array(
            [agent.pose.position.x, agent.pose.position.y, agent.pose.position.z])

        self.broadcaster.sendTransform(
            (agent.pose.position.x,
             agent.pose.position.y, agent.pose.position.z),
            (agent.pose.orientation.x,
             agent.pose.orientation.y, agent.pose.orientation.z, agent.pose.orientation.w),
            rospy.Time.now(),
            self.fram_name,
            self.paraent_name
        )

        # self.broadcaster.sendTransform((0, 0, 0), (0, 0, 0, 1), rospy.Time.now(), "odom", self.paraent_name)

        pose_msg = PoseStamped()
        pose_msg.header.stamp = rospy.Time.now()
        pose_msg.header.frame_id = self.paraent_name
        pose_msg.pose.position.x = agent.pose.position.x
        pose_msg.pose.position.y = agent.pose.position.y
        pose_msg.pose.position.z = agent.pose.position.z
        pose_msg.pose.orientation.x = agent.pose.orientation.x
        pose_msg.pose.orientation.y = agent.pose.orientation.y
        pose_msg.pose.orientation.z = agent.pose.orientation.z
        pose_msg.pose.orientation.w = agent.pose.orientation.w


        self.pub_pose.publish(pose_msg)

        odom= Odometry()
        odom.header.stamp= rospy.Time.now()
        odom.header.frame_id = "map"
        odom.pose.pose.position.x = agent.pose.position.x
        odom.pose.pose.position.y = agent.pose.position.y
        odom.pose.pose.position.z = agent.pose.position.z
        odom.pose.pose.orientation.x = agent.pose.orientation.x
        odom.pose.pose.orientation.y = agent.pose.orientation.y
        odom.pose.pose.orientation.z = agent.pose.orientation.z
        odom.pose.pose.orientation.w = agent.pose.orientation.w
        self.pub_odometry.publish(odom)



if __name__ == "__main__":
    rospy.init_node("wheel_tf")
    wheeltf = WheelTF()
    while not rospy.is_shutdown():
        wheeltf.gazebo_odom()
        rospy.sleep(0.1)