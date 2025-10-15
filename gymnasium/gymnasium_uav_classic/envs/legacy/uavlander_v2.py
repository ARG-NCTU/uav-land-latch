import gymnasium as gym
from typing import Optional

from gymnasium import error, spaces, utils
# from gymnasium.utils import seeding
import rospy
import time
import numpy as np
import math
import random
import sys
import os
import queue
from matplotlib import pyplot as plt
from sensor_msgs.msg import LaserScan, Imu
from std_srvs.srv import Empty
from std_msgs.msg import Int64, Header
from geometry_msgs.msg import Twist, Vector3, PoseStamped
from gazebo_msgs.msg import ModelState, ContactsState
from gazebo_msgs.srv import SetModelState, GetModelState, GetPhysicsProperties, SetPhysicsProperties, SetPhysicsPropertiesRequest
from scipy.spatial.transform import Rotation as R
from gazebo_msgs.srv import ApplyBodyWrench
from geometry_msgs.msg import Wrench, Point
import csv
from gymnasium_vrx.utils import (
    FixedQueue,
    SJTUROSController,
    UWBROSConnector,
)
from gymnasium_vrx.utils.uav_gazebo_ros_connector import GazeboROSConnector
from tf.transformations import quaternion_from_euler
from gymnasium.utils import seeding

class CircularQueue:
    def __init__(self, dim:tuple):
        self.queue = np.zeros(dim, dtype=np.float32)
    def push(self, data):
        data = np.asarray(data)
        data_length = data.shape[0]
        self.queue = np.roll(self.queue, data_length)
        self.queue[:data_length] = data
    def clear(self):
        self.queue[:] = 0
    def get(self):
        return self.queue
class UavLanderV2(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode: Optional[str] = None, seed=0):


        self.uwb_hist_len = 3
        self.uwb_frame_len = 6
        self.action_len = 5
        self.uav_act_hist_len = 3
        ################GYM###############
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_len,), dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Dict({
            'uwb_ranges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.uwb_hist_len, self.uwb_frame_len), dtype=np.float32, seed=seed),
            'uav_action': gym.spaces.Box(low=-1, high=1, shape=(self.uav_act_hist_len, self.action_len), dtype=np.float32, seed=seed),
        })

        self.goal = np.zeros(4) # x, y, z, yaw
        self.counter_step = 0
        self.counter_total_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0
        self.goal_range = None
        self.goal_pose = None
        self.uav_act = None
        self.game_over = False
        self.prev_shaping = None
        self.prev_r = 0.0
        self.prev_h = 0.0
        self.init_distance = None
        self.land_cnt = 0
        ################ROS#################
        rospy.init_node("gym_drone_range_" + str(seed))

        self.goal_pub = rospy.Publisher("/move_base_simple/goal/{}".format(seed), PoseStamped, queue_size=10)

        self.pos_goal_to_uav = np.zeros((4,))
        self.range = np.zeros((self.uwb_frame_len,))

        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        
        self.uav = SJTUROSController(model_name="drone".format(seed), cont_freq=20.0)
        self.uwb = UWBROSConnector(model_name="drone".format(seed), tag_num=1)
        self.gazebo = GazeboROSConnector()
        self.uwb_hist = CircularQueue(dim=(self.uwb_hist_len, self.uwb_frame_len))
        self.uav_act_hist = CircularQueue(dim=(self.uav_act_hist_len, self.action_len))
        self.gazebo.set_physic_properties()

    def step(self, action):
        self.counter_step += 1
        self.counter_total_step += 1
        self.uav_act = action
        if action[4] <= 0:
            self.uav.takeoff()
            self._move(action[0], action[1], action[2], action[3])
        

        state = self._get_observation()
        reward, terminated, truncated = self._get_reward(action, self.counter_step)
        self.episode_reward += reward
        info = self._get_info()
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.gazebo.pause()
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.counter_step = 0
        self.counter_episode += 1
        self.episode_reward = 0.0
        self.uwb_hist.clear()
        self.uav_act_hist.clear()
        self.game_over = False
        self.prev_shaping = None
        self.prev_r = 0.0
        self.prev_h = 0.0
        self.land_cnt = 0
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self._get_initial_state('wamv'))
            self.reset_model(self._get_initial_state('drone'))
        except(rospy.ServiceException) as e:
            print(e)
        # goal_pose=np.array([0, 0, 0.26, 0])
        self.goal_range = self.uwb.cal_drone_pose_to_uwb_range(np.array([0, 0, 0.26, 0]))
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        # if self.counter_episode == 1:
        obs = self._get_observation()
        self.init_distance = np.linalg.norm(self.range - self.goal_range)
        self.gazebo.unpause()
        return obs, self._get_info()

    def close(self):
        self._move(0, 0, 0, 0)

############################################################################################################

    def _get_reward(self, action, step):
        terminated = False
        truncated = False
        reward = 0.0
        max_episode_steps = 1024
        reward_max = 100

        goal_z = self.gazebo.wamv_pose[2]+2 # 2m above wamv's bottom
        goal_z_tol = 0.2 # 20cm

        r = np.linalg.norm(self.gazebo.uav_pose[:1] - self.gazebo.wamv_pose[:1])
        h = self.gazebo.uav_pose[2] - goal_z

        u = lambda t: np.where(t >= 0, 1, 0)
        curve = lambda x: 2 / (1 + math.exp(x))

        hight_score = -reward_max*curve(abs(h+goal_z))*(1-u(h+goal_z_tol))+reward_max*u(goal_z_tol-abs(h)) + reward_max*curve(h*0.8)*u(h-goal_z_tol)
        #------------- under platform ------------------------------------- between platform +- 5cm -------------------- above platform
        reward = hight_score*curve(r)
        reward += (
            - 5  * abs(self.uav_act[2]) if self.uav_act[2] > 0 and (u(h-self.prev_h) and u(h+goal_z_tol)) else 0
            + 50 * abs(self.uav_act[2]) if (1-u(h+goal_z_tol)) and self.uav_act[2] > 0 else 0
            - 1 * abs(self.uav_act[3])
            - 5 * math.sqrt(self.uav_act[0]**2 + self.uav_act[1]**2) if u(r - self.prev_r) and u(h+goal_z_tol) else 0
            + 20 * curve(r) if u(self.prev_r-r) and u(h+goal_z_tol) else 0
            + 2 * math.sqrt(self.uav_act[0]**2 + self.uav_act[1]**2) if u(r - self.prev_r) and (1-u(h+goal_z_tol)) else 0
        )
        self.prev_r = r
        self.prev_h = h
        if self.uav_act[4] > 0:# landed act
            if abs(self.gazebo.uav_pose[2] - goal_z) <= goal_z_tol and abs(r) <= 0.3:
                self.land_cnt += 1    
                reward = self.land_cnt*reward_max*curve(abs(r))
                if self.land_cnt > 3:
                    terminated = True
                    reward = 10*reward_max*curve(abs(r))
            else:
                reward -= reward_max/5
                self.land_cnt = 0
        else:
            if self.counter_step >= max_episode_steps:
                terminated = True
                reward = -100
        if self.uav.get_pose()[2] <= 0.0:# touch water
            terminated = True
            reward = -10*reward_max

        output = "\rstep:{:3d}, actions[{},{},{},{},{}], r: {:.2f}, h:{:.2f} hight_score {:.2f} reward:{}".format(
            self.counter_step,
            " {:.2f}".format(self.uav_act[0]) if self.uav_act[0] >= 0 else "{:.2f}".format(self.uav_act[0]),
            " {:.2f}".format(self.uav_act[1]) if self.uav_act[1] >= 0 else "{:.2f}".format(self.uav_act[1]),
            " {:.2f}".format(self.uav_act[2]) if self.uav_act[2] >= 0 else "{:.2f}".format(self.uav_act[2]),
            " {:.2f}".format(self.uav_act[3]) if self.uav_act[3] >= 0 else "{:.2f}".format(self.uav_act[3]),
            " {:.2f}".format(self.uav_act[4]) if self.uav_act[4] >= 0 else "{:.2f}".format(self.uav_act[4]),
            r, h,            
            hight_score,
            " {:.2f}".format(reward) if reward >= 0 else "{:.2f}".format(reward)
        )

        sys.stdout.write(output)
        sys.stdout.flush()
        reward /= 1000
        return reward, terminated, truncated
    
    def _get_observation(self):
        # self.prev_r = np.linalg.norm(self.range - self.goal_range)
        self.range = self.uwb.get_range() 
        self.uwb_hist.push(self.range)
        if self.uav_act is not None:
            self.uav_act_hist.push(self.uav_act)
        return {'uwb_ranges': self.uwb_hist.get(), 'uav_action': self.uav_act_hist.get()}

    def _move(self, x, y, z, yaw):
        self.uav.move(x, y, z, yaw)

            # self._move(0, 0, 0, 0)
        # self._move(0,0,0,0)

    def _get_info(self):
        return {}
    
    def _get_initial_state(self, name):
        # start position
        state_msg = ModelState()
        state_msg.model_name = name
        x_rand = y_rand = 0 
        # state_msg.pose.position.x = 0
        # state_msg.pose.position.y = 0
        if name == 'wamv':
            state_msg.pose.position.z = 0
        elif name == 'drone':
            self.uav.takeoff()
            state_msg.pose.position.z = np.random.uniform(4, 6)
            x_rand = np.random.uniform(-2, 2)
            y_rand = np.random.uniform(-2, 2)

        state_msg.pose.position.x = x_rand
        state_msg.pose.position.y = y_rand
        # angle = 1.57
        angle_rand = np.random.uniform(-0.35, 0.35)
        angle = 1.57 + angle_rand
        if angle >= np.pi:
            angle -= 2*np.pi
        elif angle <= -np.pi:
            angle += 2*np.pi
        r = R.from_euler('z', angle)
        quat = r.as_quat()
        state_msg.pose.orientation.x = quat[0]
        state_msg.pose.orientation.y = quat[1]
        state_msg.pose.orientation.z = quat[2]
        state_msg.pose.orientation.w = quat[3]
        return state_msg

