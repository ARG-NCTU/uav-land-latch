import math
from typing import Optional

import numpy as np
import rospy
import yaml
from geometry_msgs.msg import PoseStamped
from gymnasium_vrx.utils import (
    FixedQueue,
    GazeboROSConnector,
    SJTUROSController,
    UWBROSConnector,
)
from tf.transformations import quaternion_from_euler

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding


class DroneRangeYZV1GYM(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, seed=0):
        obs_hist_len = 1
        obs_frame_len = 6
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64, seed=seed)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_hist_len * obs_frame_len + obs_frame_len,), dtype=np.float64, seed=seed
        )

        self.goal = np.zeros(4)
        self.env_ros = DroneRangeYZV1ROS(
            seed, obs_hist_len=obs_hist_len, obs_frame_len=obs_frame_len, action_len=2, t_tol=0.5, r_tol=0.1
        )
        self.counter_step = 0
        self.counter_total_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0

    def step(self, action):
        self.counter_step += 1
        self.counter_total_step += 1
        self.env_ros.move(0.0, action[0], action[1], 0.0)
        state = self.env_ros.get_observation()
        reward, terminated, truncated = self.env_ros.get_reward(action, self.counter_step)
        self.episode_reward += reward
        info = self.env_ros.get_info()
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.counter_step = 0
        self.counter_episode += 1
        self.episode_reward = 0.0

        goal_yz_range = 5.0
        self.goal = np.zeros(4)
        self.goal[1] = self._np_random.uniform(-goal_yz_range, goal_yz_range)
        self.goal[2] = self._np_random.uniform(-goal_yz_range, goal_yz_range)

        # step = 0.1
        # y_grid = np.arange(-goal_yz_range, goal_yz_range + step, step)
        # z_grid = np.arange(-goal_yz_range, goal_yz_range + step, step)
        # y = np.random.choice(y_grid)
        # z = np.random.choice(z_grid)
        # self.goal[1] = y
        # self.goal[2] = z

        offset = 0.5
        offset_y = self._np_random.uniform(-offset, offset)
        offset_z = self._np_random.uniform(-offset, offset)

        # drone_pose_y = np.clip(self.goal[1] + offset_y, 0, goal_yz_range)
        # drone_pose_z = np.clip(self.goal[2] + offset_z, 0, goal_yz_range)

        drone_pose = np.zeros(4)
        drone_pose[1] = self.goal[1] + offset_y
        drone_pose[2] = self.goal[2] + offset_z
        drone_pose[1] = self._np_random.uniform(-goal_yz_range, goal_yz_range)
        drone_pose[2] = self._np_random.uniform(-goal_yz_range, goal_yz_range)

        # drone_pose = np.array([0.0, 0.0, 1.0, 0.0])
        # self.goal = np.array([0.0, 0.0, 1.0, 0.0])
        # print("seed: ", seed)
        # print("drone_pose: ", drone_pose)
        # print("goal: ", self.goal)

        self.env_ros.move(0, 0, 0, 0)
        self.env_ros.reset(drone_pose=drone_pose, goal_pose=self.goal)
        self.env_ros.move(0, 0, 0, 0)
        self.env_ros.move(0, 0, 0, 0)
        self.env_ros.move(0, 0, 0, 0)
        self.env_ros.move(0, 0, 0, 0)
        if self.counter_episode == 1:
            self.env_ros.drone.takeoff()
        obs = self.env_ros.get_observation()
        return obs, self.env_ros.get_info()

    def close(self):
        self.env_ros.move(0, 0, 0, 0)
        self.env_ros.reset()


class DroneRangeYZV1ROS:
    def __init__(self, seed=0, obs_hist_len=10, obs_frame_len=12, action_len=4, t_tol=0.5, r_tol=0.1):
        rospy.init_node("gym_drone_range_" + str(seed))

        self.goal_pub = rospy.Publisher("/move_base_simple/goal/{}".format(seed), PoseStamped, queue_size=10)

        self.obs_hist_len = obs_hist_len
        self.obs_frame_len = obs_frame_len
        self.action_len = action_len

        self.t_tol = t_tol
        self.r_tol = r_tol

        self.pose = np.zeros((4,))
        self.pos_goal_to_drone = np.zeros((4,))
        self.range = np.zeros((obs_frame_len,))

        self.goal_pose = np.zeros_like(self.pose)
        self.goal_range = np.zeros_like(self.range)

        self.drone = SJTUROSController(model_name="drone{}".format(seed), cont_freq=20.0)
        self.uwb = UWBROSConnector(model_name="drone{}".format(seed), tag_num=1)
        self.gazebo = GazeboROSConnector()
        self.history = FixedQueue(obs_hist_len * obs_frame_len)

        self.gazebo.set_physic_properties()

    def get_reward(self, action, step):
        terminated = False
        truncated = False
        reward = 0
        max_episode_steps = 512

        # Reward 1: Euclidean distance of signature
        dist = np.linalg.norm(self.range - self.goal_range)
        reward += np.exp(-np.power(dist, 2))
        reward /= max_episode_steps

        # Reward 2: Cosine similarity of signature
        # drone_range_norm = self.range / 10.0
        # goal_range_norm = self.goal_range / 10.0
        # cosine_sim = np.dot(drone_range_norm, goal_range_norm) / (
        #     np.linalg.norm(drone_range_norm) * np.linalg.norm(goal_range_norm)
        # )
        # print("cosine_sim: ", cosine_sim)
        # reward += cosine_sim
        # reward /= max_episode_steps

        # Reward 3: Euclidean distance of pose
        # dist = np.linalg.norm(self.pose - self.goal_pose)
        # reward += np.exp(-np.power(dist, 2))
        # reward /= max_episode_steps

        # if dist > 5:
        #     truncated = True

        return reward, terminated, truncated

    def get_observation(self):
        self.pose = self.drone.get_pose()
        self.range = self.uwb.get_range()
        self.history.push(self.range)
        observation = np.concatenate([self.goal_range, self.history.get()])
        return observation

    def move(self, x, y, z, yaw):
        self.drone.move(x, y, z, yaw)

    def reset(self, drone_pose=np.zeros((4,)), goal_pose=np.zeros((4,))):
        self.history.clear()
        self.goal_pose = goal_pose
        self.goal_range = self.uwb.cal_drone_pose_to_uwb_range(goal_pose)
        self.drone.reset(x=drone_pose[0], y=drone_pose[1], z=drone_pose[2], yaw=drone_pose[3])
        # Publish goal
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        goal_msg.pose.position.x = goal_pose[0]
        goal_msg.pose.position.y = goal_pose[1]
        goal_msg.pose.position.z = goal_pose[2]
        quaternion = quaternion_from_euler(0, 0, goal_pose[3])
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]
        self.goal_pub.publish(goal_msg)

    def get_info(self):
        return {}


class DroneRangeYZSimpleV0(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, seed=0):
        self.max_speed = 1.0
        self.max_angular_speed = 1.0
        self.dt = 0.05

        action_high = np.array(
            [self.max_speed, self.max_speed, self.max_speed, self.max_angular_speed], dtype=np.float64
        )

        action_high = np.array([self.max_speed, self.max_speed], dtype=np.float64)
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float64, seed=seed)

        observation_high = np.full((6 + 6,), np.inf, dtype=np.float64)
        self.observation_space = spaces.Box(low=-observation_high, high=observation_high, dtype=np.float64)

        self.drone_pose = np.zeros(4, dtype=np.float64)  # x, y, z, yaw
        self.drone_velocity = np.zeros(4, dtype=np.float64)  # vx, vy, vz, vyaw

        self.goal_pose = np.zeros(4, dtype=np.float64)  # x, y, z, yaw

        self.drone_signature = np.zeros(6, dtype=np.float64)
        self.goal_signature = np.zeros(6, dtype=np.float64)

        self.counter_step = 0
        self.counter_total_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0

        self.respawn_range = 7.0

        yaml_file_path = "/home/argrobotx/robotx-2022/catkin_ws/src/pozyx_ros/config/wamv/wamv.yaml"
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        self.uwb_anchor_pos = np.array([[item["x"], item["y"], item["z"]] for item in data.values()]) / 1000.0
        # print("Anchor poses: \n{}\n".format(self.uwb_anchor_pos))

        yaml_file_path = "/home/argrobotx/robotx-2022/catkin_ws/src/pozyx_ros/config/wamv/drone.1.yaml"
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        self.uwb_tag_pos = np.array([[item["x"], item["y"], item["z"]] for item in data.values()]) / 1000.0
        # print("Tag poses: \n{}\n".format(self.uwb_tag_pos))
        print(f"Environment initialized: {seed}")

    def step(self, action):
        self.counter_step += 1
        self.counter_total_step += 1
        # self.drone_velocity[1:3] = action
        # Calculate new position considering yaw
        # yaw = self.drone_pose[3]

        # dy, dz = self.drone_velocity[1:3] * self.dt
        dy, dz = action * self.dt

        # Rotate dx and dy according to the yaw angle
        # cos_yaw = math.cos(yaw)
        # sin_yaw = math.sin(yaw)

        # dx_rotated = cos_yaw * dx - sin_yaw * dy
        # dy_rotated = sin_yaw * dx + cos_yaw * dy

        # Update yaw
        # yaw = angle_normalize(yaw + dyaw)

        # Update position
        # self.drone_pose[0] += dx_rotated
        # self.drone_pose[1] += dy_rotated
        self.drone_pose[1] += dy
        self.drone_pose[2] += dz
        # self.drone_pose[3] = yaw

        # Convert pose to signature
        observation = np.concatenate([self.goal_signature, self.pose_to_signature(self.drone_pose)])

        dist = np.linalg.norm(self.pose_to_signature(self.drone_pose) - self.goal_signature)
        reward = np.exp(-(dist**2)) / 1024
        # reward = np.exp(-(dist**2)) * self.counter_step / 524800
        # terminated = False
        # truncated = False
        # info = {}
        self.episode_reward += reward
        # return observation, reward, terminated, truncated, info
        return observation, reward, False, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        self.counter_step = 0
        self.counter_episode += 1
        self.episode_reward = 0.0
        super().reset(seed=seed)
        # Random reset drone pose and goal pose
        # ranges = [
        #     ((0e6 / 70.0, 2e6 / 70.0), 2.0),
        #     ((2e6 / 70.0, 7e6 / 70.0), 5.0),
        #     ((7e6 / 70.0, 15e6 / 70.0), 7.0),
        #     ((15e6 / 70.0, 30e6 / 70.0), 10.0),
        # ]
        # for range_, value in ranges:
        #     start, end = range_
        #     if start <= self.counter_total_step <= end:
        #         self.respawn_range = value
        #         break
        # else:
        #     self.respawn_range = 10
        # self.respawn_range = 5.0

        self.goal_pose[1:3] = self.np_random.uniform(-self.respawn_range, self.respawn_range, size=2)
        offset = self.np_random.uniform(-3, 3, size=2)
        self.drone_pose[1:3] = self.goal_pose[1:3] + offset
        # Convert pose to signature
        self.goal_signature = self.pose_to_signature(self.goal_pose)

        observation = np.concatenate([self.goal_signature, self.pose_to_signature(self.drone_pose)])
        return observation, {}

    def pose_to_signature(self, pose: np.ndarray) -> np.ndarray:
        # Calculate signature from pose
        signature = np.zeros(6, dtype=np.float64)
        # Transform pose to uwb tag pos
        uwb_tag_pos = self.body_pose_to_uwb_tag_pos(pose)
        # Calculate signature
        distances = np.linalg.norm(uwb_tag_pos[:, None, :] - self.uwb_anchor_pos[None, :, :], axis=2)
        signature = np.min(distances, axis=0)

        return signature

    def body_pose_to_uwb_tag_pos(self, pose: np.ndarray) -> np.ndarray:
        # self.uwb_tag_pos is the position of the uwb tag in the body frame
        # self.uwb_tag_pos is like [[x, y, z], [x, y, z], ...]
        # This function should return the position of the uwb tag in the world frame
        assert pose.shape == (4,)
        x, y, z, yaw = pose
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
        transformation_matrix = np.array([[cos_yaw, -sin_yaw, x], [sin_yaw, cos_yaw, y], [0, 0, 1]])
        uwb_tag_pos_homogeneous = np.hstack((self.uwb_tag_pos[:, :2], np.ones((self.uwb_tag_pos.shape[0], 1))))
        tag_pos_transformed_2d = np.dot(uwb_tag_pos_homogeneous, transformation_matrix.T)
        tag_pos_transformed = np.hstack((tag_pos_transformed_2d[:, :2], np.full((self.uwb_tag_pos.shape[0], 1), z)))
        assert tag_pos_transformed.shape == self.uwb_tag_pos.shape
        return tag_pos_transformed


# def angle_normalize(x):
#     return ((x + np.pi) % (2 * np.pi)) - np.pi
