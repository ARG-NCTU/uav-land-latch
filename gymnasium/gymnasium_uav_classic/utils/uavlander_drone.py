import numpy as np
import rospy
import yaml
from std_msgs.msg import Float64MultiArray
import time
from gazebo_msgs.msg import ModelState
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import Empty
from tf.transformations import euler_from_quaternion, quaternion_from_euler


class uavROSConnector:
    def __init__(self, model_name="sjtu_drone", tags_num=2) -> None:
        self.uwb = UWBROSConnector(model_name=model_name, tag_num=tags_num)
        self.gazebo_pose = np.zeros(4)
        self.ctrler = SJTUROSController(model_name='drone', cont_freq=20.0)
        rospy.Subscriber(
                    "/gazebo/sjtu_drone/pose_stamped",
                    PoseStamped,
                    self.cb_uav_pose,
                    queue_size=10,
                )
        
    def cb_uav_pose(self, msg):
        theta_z = euler_from_quaternion(
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            )
        )[2]
        self.gazebo_pose = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z, theta_z])


class UWBROSConnector:
    def __init__(self, model_name="sjtu_drone", tag_num=2, anchor_num=6):
        self.model_name = model_name
        self.tag_num = tag_num
        self.anchor_num = anchor_num

        self.uwb_range = np.zeros(tag_num * anchor_num, dtype=np.float64)
        self.uwb_anchor_pose = np.zeros((anchor_num, 3))
        self.uwb_tag_pose = np.zeros((tag_num, 3))

        self.uwb_sub = []
        for i in range(tag_num):
            self.uwb_sub.append(
                rospy.Subscriber(
                    "/pozyx_simulation/uwb{}/distances".format(i),
                    Float64MultiArray,
                    self.update_uwb_range,
                    i * anchor_num,
                )
            )
        yaml_file_path = "/home/argrobotx/uav-land-latch/catkin_ws/src/pozyx_ros/config/wamv/wamv.yaml"
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)

        coordinates_list = []
        for item in data.values():
            coordinates_list.append([item["x"], item["y"], item["z"]])
        self.uwb_anchor_pose = np.array(coordinates_list) / 1000.0
        print("Anchor poses: \n{}\n".format(self.uwb_anchor_pose))

        yaml_file_path = "/home/argrobotx/uav-land-latch/catkin_ws/src/pozyx_ros/config/wamv/drone.yaml"
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)

        coordinates_list = []
        for item in data.values():
            coordinates_list.append([item["x"], item["y"], item["z"]])
        self.uwb_tag_pose = np.array(coordinates_list) / 1000.0
        print("Tag poses: \n{}\n".format(self.uwb_tag_pose))

    def update_uwb_range(self, msg, offset):
        self.uwb_range[offset : offset + 6] = msg.data[:6]

    def get_range(self):
        return self.uwb_range

    def cal_drone_pose_to_uwb_range(self, drone_pose):
        assert drone_pose.shape == (4,)

        # Extract drone position and yaw angle
        x, y, z, theta = drone_pose
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        # 4x4 Transformation matrix from drone frame to world frame
        transformation_matrix = np.array(
            [[cos_theta, -sin_theta, 0, x], [sin_theta, cos_theta, 0, y], [0, 0, 1, z], [0, 0, 0, 1]]
        )

        # Calculate distances
        distances = []
        for tag in self.uwb_tag_pose:
            # Transform tag position to world frame (using homogeneous coordinates)
            tag_homogeneous = np.append(tag, 1)
            tag_world = np.dot(transformation_matrix, tag_homogeneous)[:3]
            # print("tag_world: ", tag_world)

            for anchor in self.uwb_anchor_pose:
                # Euclidean distance between tag and anchor
                distance = np.linalg.norm(tag_world - anchor)
                distances.append(distance)

        return np.array(distances)

    def calc_distance(self, target_points):
        distances = np.zeros((target_points.shape[0], self.uwb_anchor_pose.shape[0]))
        for i, target in enumerate(target_points):
            distances[i] = np.sqrt(np.sum((self.uwb_anchor_pose - target) ** 2, axis=1))
        return distances


class SJTUROSController:
    def __init__(self, model_name="sjtu_drone", cont_freq=20.0, reset_method="service"):
        self.model_name = model_name
        self.cont_freq = cont_freq
        self.reset_method = reset_method.lower()

        self.rate = rospy.Rate(cont_freq)

        self.pose = PoseStamped()
        self.cmd_vel = Twist()

        # Drone Publisher and Subscriber
        self.pose_sub = rospy.Subscriber("/gazebo/{}/pose_stamped".format(model_name), PoseStamped, self.cb_pose)
        self.pub_twist = rospy.Publisher("/{}/cmd_vel".format(model_name), Twist, queue_size=10)
        self.pub_reset = rospy.Publisher("/{}/reset".format(model_name), Empty, queue_size=10)
        self.pub_takeoff = rospy.Publisher("/{}/takeoff".format(model_name), Empty, queue_size=10)

        self.pub_gazebo_set_model_state = rospy.Publisher("/gazebo/set_model_state", ModelState, queue_size=10)
        self.srv_gazebo_set_model_state = rospy.ServiceProxy("/gazebo/set_model_state", SetModelState)
        # self.move_timer = rospy.Timer(rospy.Duration(1.0 / self.cont_freq), self.cb_move_timer)
        while self.pub_gazebo_set_model_state.get_num_connections() == 0:
            rospy.logwarn_throttle(1, "Waiting for gazebo/set_model_state to be subscribed")

    def cb_pose(self, msg):
        self.pose = msg

    def get_pose(self):
        theta_z = euler_from_quaternion(
            (
                self.pose.pose.orientation.x,
                self.pose.pose.orientation.y,
                self.pose.pose.orientation.z,
                self.pose.pose.orientation.w,
            )
        )[2]
        return np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z, theta_z])

    def reset_random(self, r_min=0.0, r_max=1.0, z_min=9.0, z_max=11.0, yaw_dist=0.2 * np.pi):
        print("Resetting the drone...")
        self.move(0, 0, 0, 0, immediate=True)
        model_state = ModelState()

        # random pose
        r = np.random.uniform(r_min, r_max)
        theta = np.random.uniform(0, 2 * np.pi)

        model_state.model_name = self.model_name
        model_state.pose.position.x = r * np.cos(theta)
        model_state.pose.position.y = r * np.sin(theta)
        model_state.pose.position.z = np.random.uniform(z_min, z_max)

        # random orientation
        yaw = np.random.uniform(-yaw_dist, yaw_dist)
        quaternion = quaternion_from_euler(0, 0, yaw)
        model_state.pose.orientation.x = quaternion[0]
        model_state.pose.orientation.y = quaternion[1]
        model_state.pose.orientation.z = quaternion[2]
        model_state.pose.orientation.w = quaternion[3]

        model_state.twist.linear.x = 0.0
        model_state.twist.linear.y = 0.0
        model_state.twist.linear.z = 0.0
        model_state.twist.angular.x = 0.0
        model_state.twist.angular.y = 0.0
        model_state.twist.angular.z = 0.0
        model_state.reference_frame = "map"

        model_state_msg = SetModelStateRequest()
        model_state_msg.model_state = model_state

        self.move(0, 0, 0, 0, immediate=True)

        if self.reset_method == "topic":
            print("Resetting the drone using publisher")
            self.pub_gazebo_set_model_state.publish(model_state)
        elif self.reset_method == "service":
            print("Resetting the drone using service")
            try:
                self.srv_gazebo_set_model_state(model_state_msg)
            except rospy.ServiceException as e:
                print("Service call failed: %s" % e)
                self.pub_gazebo_set_model_state.publish(model_state)

        self.move(0, 0, 0, 0, immediate=True)
        self.pub_reset.publish()
        self.move(0, 0, 0, 0, immediate=True)
        # time.sleep(0.5)

    def reset(self, x=0.0, y=0.0, z=0.0, yaw=0.0):
        # print("Resetting the drone...")
        self.move(0, 0, 0, 0, immediate=True)
        model_state = ModelState()
        model_state.model_name = self.model_name
        model_state.pose.position.x = x
        model_state.pose.position.y = y
        model_state.pose.position.z = z
        quaternion = quaternion_from_euler(0, 0, yaw)
        model_state.pose.orientation.z = quaternion[2]
        model_state.pose.orientation.w = quaternion[3]

        if self.reset_method == "topic":
            # print("Resetting the drone using publisher")
            self.pub_gazebo_set_model_state.publish(model_state)
        elif self.reset_method == "service":
            # print("Resetting the drone using service")
            model_state_req = SetModelStateRequest()
            model_state_req.model_state = model_state
            try:
                self.srv_gazebo_set_model_state(model_state_req)
            except rospy.service.ServiceException as e:
                print("Service call failed: %s" % e)
                self.pub_gazebo_set_model_state.publish(model_state)
        self.pub_reset.publish()
        # time.sleep(0.5)
        self.move(0, 0, 0, 0, immediate=True)

    def takeoff(self):
        self.pub_takeoff.publish()

    def move(self, vx, vy, vz, wz, immediate=False):
        self.cmd_vel.linear.x = vx
        self.cmd_vel.linear.y = vy
        self.cmd_vel.linear.z = vz
        self.cmd_vel.angular.z = wz
        if immediate:
            self.pub_twist.publish(self.cmd_vel)
            return
        self.pub_twist.publish(self.cmd_vel)
        try:
            self.rate.sleep()
        except rospy.ROSTimeMovedBackwardsException as e:
            rospy.logwarn("ROSTimeMovedBackwardsException: {}".format(e))

    def cb_move_timer(self, event):
        self.pub_twist.publish(self.cmd_vel)