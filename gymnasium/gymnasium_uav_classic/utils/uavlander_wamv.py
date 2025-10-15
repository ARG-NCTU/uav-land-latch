import rospy
import numpy as np
from gazebo_msgs.msg import ODEPhysics
from gazebo_msgs.srv import SetPhysicsProperties, SetPhysicsPropertiesRequest
from geometry_msgs.msg import Vector3
from std_srvs.srv import Empty as EmptySrv
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import PoseStamped
from tf.transformations import euler_from_quaternion, quaternion_from_euler

class wamvGazeboROSConnector:

    def __init__(self):
        self.gazebo_pose = np.zeros(4)
        # self.gazebo_pose = None
        rospy.Subscriber(
                    "/gazebo/wamv/pose_stamped",
                    PoseStamped,
                    self.cb_wamv_pose,
                    queue_size=10,
                )

    def cb_wamv_pose(self, msg):
            theta_z = euler_from_quaternion(
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            )
        )[2]
            self.gazebo_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, theta_z])