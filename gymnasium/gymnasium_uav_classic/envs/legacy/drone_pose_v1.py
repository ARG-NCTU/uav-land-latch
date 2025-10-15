import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped
from gymnasium_vrx.utils import FixedQueue, GazeboROSConnector, SJTUROSController
from scipy.spatial.transform import Rotation as R
from tf.transformations import quaternion_from_euler

import gymnasium as gym
from gymnasium.utils import seeding


class DronePoseV1GYM(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, seed=0):
        obs_hist_len = 2
        obs_frame_len = 4
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_hist_len * obs_frame_len,), dtype=np.float64
        )

        self.goal = np.zeros(4)
        self.env_ros = DronePoseV1ROS(
            seed, obs_hist_len=obs_hist_len, obs_frame_len=obs_frame_len, action_len=4, t_tol=0.5, r_tol=0.1
        )
        self.counter_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0

    def step(self, action):
        # self.env_ros.gazebo.unpause()
        self.counter_step += 1
        self.env_ros.move(action[0], action[1], action[2], action[3])
        state = self.env_ros.get_observation()
        reward, terminated, truncated = self.env_ros.get_reward(action)
        self.episode_reward += reward
        info = self.env_ros.get_info()
        # self.env_ros.gazebo.pause()
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)

        self.counter_episode += 1
        self.episode_reward = 0.0

        drone_pose = np.empty(4)
        drone_pose[:3] = self._np_random.uniform(-0, 0, size=(3,))
        drone_pose[3] = self._np_random.uniform(-np.pi, np.pi)
        # drone_pose[3] = self._np_random.uniform(0, 0)

        self.goal = np.empty(4)
        self.goal[:3] = self._np_random.uniform(-0, 0, size=(3,))
        self.goal[3] = self._np_random.uniform(-np.pi, np.pi)
        # self.goal[3] = self._np_random.uniform(0, 0)

        self.env_ros.gazebo.unpause()
        self.env_ros.reset(self.counter_step, drone_pose=drone_pose, goal_pose=self.goal)
        if self.counter_episode == 1:
            self.env_ros.drone.takeoff()
        obs = self.env_ros.get_observation()
        return obs, self.env_ros.get_info()

    def close(self):
        self.env_ros.move(0, 0, 0, 0)
        self.env_ros.reset(step=None)


class DronePoseV1ROS:
    def __init__(self, seed=0, obs_hist_len=10, obs_frame_len=4, action_len=4, t_tol=0.5, r_tol=0.1):
        rospy.init_node("gym_drone_pose_" + str(seed))

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
        self.gazebo = GazeboROSConnector()
        self.observation = FixedQueue(obs_hist_len * obs_frame_len)

        self.r_min = 0.1
        self.r_max = 10.0

        self.r_min_scheduler = self.create_scheduler(self.r_min, 3e6, 8e6, 10.0)
        self.r_max_scheduler = self.create_scheduler(self.r_max, 3e6, 8e6, 50.0)

        self.gazebo.set_physic_properties()

    def get_reward(self, action):
        reward = 1.0
        terminated = False

        position_error = np.linalg.norm(self.pos_goal_to_drone[:3])
        angle_error = np.linalg.norm(self.pose[3] - self.goal_pose[3])

        normalized_linear_action_error = np.square(np.linalg.norm(action[:3])) / np.maximum(position_error, self.t_tol)
        normalized_angular_action_error = np.square(action[3]) / np.maximum(angle_error, self.r_tol)

        reward += (
            -1.0 * position_error
            - 1.0 * angle_error
            # - 0.015 * normalized_linear_action_error
            # - 0.002 * normalized_angular_action_error
            # + 0.2 * cosine_similarity
        )

        # reward += -error
        reward *= 1e-2

        # print("pose: ", self.pos_goal_to_drone[:3])
        # print("reward: ", reward)

        truncated = False
        if np.linalg.norm(self.pose[:3]) > 100.0:
            truncated = True
            reward -= 1000.0
            print("truncated")
        return reward, terminated, truncated

    def get_observation(self):
        self.pose = self.drone.get_pose()
        tm_drone_to_target = np.zeros((4, 4))
        tm_drone_to_target[0:3, 3] = self.pose[0:3]
        tm_drone_to_target[3, 3] = 1
        tm_drone_to_target[0:3, 0:3] = R.from_euler("z", self.pose[3]).as_matrix()
        tm_goal_to_target = np.zeros((4, 4))
        tm_goal_to_target[0:3, 3] = self.goal_pose[0:3]
        tm_goal_to_target[3, 3] = 1
        tm_goal_to_target[0:3, 0:3] = R.from_euler("z", self.goal_pose[3]).as_matrix()
        tm_target_to_drone = np.linalg.inv(tm_drone_to_target)
        tm_goal_to_drone = tm_target_to_drone @ tm_goal_to_target
        self.pos_goal_to_drone = tm_goal_to_drone[0:4, 3]

        obs = self.pos_goal_to_drone
        obs[3] = (obs[3] + np.pi) % (2 * np.pi) - np.pi
        self.observation.push(obs)
        return self.observation.get()

    def move(self, x, y, z, yaw):
        self.drone.move(x, y, z, yaw)

    def reset(self, step, drone_pose=np.zeros((4,)), goal_pose=np.zeros((4,))):
        # self.r_min = self.r_min_scheduler(step)
        # self.r_max = self.r_max_scheduler(step)
        # z_min = 5.0
        # z_max = 20.0
        # yaw_dist = np.pi

        self.observation.clear()
        self.goal_pose = goal_pose
        self.drone.reset(x=drone_pose[0], y=drone_pose[1], z=drone_pose[2], yaw=drone_pose[3])

        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.pose.position.x = goal_pose[0]
        goal_msg.pose.position.y = goal_pose[1]
        goal_msg.pose.position.z = goal_pose[2]
        quaternion = quaternion_from_euler(0, 0, goal_pose[3])
        goal_msg.pose.orientation.z = quaternion[2]
        goal_msg.pose.orientation.w = quaternion[3]
        self.goal_pub.publish(goal_msg)

    def create_scheduler(self, initial_value, start_steps, end_steps, max_value):
        def scheduler(current_steps):
            if current_steps < start_steps:
                return initial_value
            elif current_steps > end_steps:
                return max_value
            else:
                scale = (current_steps - start_steps) / (end_steps - start_steps)
                return initial_value + (max_value - initial_value) * scale

        return scheduler

    def get_info(self):
        return {}
