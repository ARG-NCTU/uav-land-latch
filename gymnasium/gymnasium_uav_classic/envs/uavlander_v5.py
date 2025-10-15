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
from gymnasium_uav_classic.utils.uavlander_drone import uavROSConnector
from gymnasium_uav_classic.utils.uavlander_wamv import wamvGazeboROSConnector
from gymnasium_uav_classic.utils.uavlander_gazebo import GazeboROSConnector
from gym.error import DependencyNotInstalled
from tf.transformations import quaternion_from_euler
from gymnasium.utils import seeding
from typing import TYPE_CHECKING, Optional
import pygame
VIEWPORT_W = 600
VIEWPORT_H = 400
FPS = 50
SCALE = 30.0 


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
    
class UavLanderV5(gym.Env):
    metadata = {
                "render_modes": ["rgb_array", "human"],
                "render_fps": FPS,
                }

    def __init__(self,
                 seed = 0,
                 render_mode: Optional[str] = None,
                 ):

        
        self.uwb_hist_len = 6
        self.uwb_frame_len = 12
        self.action_len = 4
        self.uav_act_hist_len = 6
        ################GYM###############
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(self.action_len,), dtype=np.float32, seed=seed)
        self.observation_space = gym.spaces.Dict({
            # 'uwb_ranges': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.uwb_hist_len, self.uwb_frame_len), dtype=np.float32, seed=seed),
            'uav_action': gym.spaces.Box(low=-1, high=1, shape=(self.uav_act_hist_len, self.action_len), dtype=np.float32, seed=seed),
            'uav_gazebo_pose': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.uav_act_hist_len, 4), dtype=np.float32, seed=seed),
        })

        self.counter_step = 0
        self.counter_total_step = 0
        self.counter_episode = 0
        self.episode_reward = 0.0
        self.uav_act = None
        self.prev_r = 0.0
        self.prev_h = 0.0
        self.land_cnt = 0

        self.render_mode = render_mode
        
        self.screen: pygame.Surface = None
        self.clock = None
        ################ROS#################
        rospy.init_node("gym_drone_range_" + str(seed))

        self.goal_pub = rospy.Publisher("/move_base_simple/goal/{}".format(seed), PoseStamped, queue_size=10)

        self.pos_goal_to_uav = np.zeros((4,))

        self.reset_model = rospy.ServiceProxy(
            '/gazebo/set_model_state', SetModelState)
        
        self.uav = uavROSConnector()
        self.wamv = wamvGazeboROSConnector()
        self.gazebo = GazeboROSConnector()
        # self.uwb_hist = CircularQueue(dim=(self.uwb_hist_len, self.uwb_frame_len))
        self.uav_pose_hist = CircularQueue(dim=(self.uav_act_hist_len, 4))
        # self.uwb1_hist = CircularQueue(dim=(self.uwb_hist_len, self.uwb_frame_len))
        self.uav_act_hist = CircularQueue(dim=(self.uav_act_hist_len, self.action_len))
        self.gazebo.set_physic_properties()

    def step(self, action):
        self.counter_step += 1
        self.counter_total_step += 1
        self.uav_act = action
        self._move(action[0], action[1], action[2], action[3])
        
        state = self._get_observation()
        reward, terminated, truncated = self._get_reward(action, self.counter_step)
        # self.episode_reward += reward
        info = self._get_info()
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):

        super().reset(seed=seed, options=options)

        self.gazebo.pause()
        if seed is not None:
            self._np_random, seed = seeding.np_random(seed)
        self.counter_step = 0
        self.counter_episode += 1
        self.episode_reward = 0.0
        # self.uwb_hist.clear()
        self.uav_pose_hist.clear()
        self.uav_act_hist.clear()
        self.prev_r = 0.0
        self.prev_h = 0.0
        self.land_cnt = 0
        rospy.wait_for_service('/gazebo/set_model_state')
        try:
            self.reset_model(self._get_initial_state('wamv'))
            self.reset_model(self._get_initial_state('sjtu_drone'))
        except(rospy.ServiceException) as e:
            print(e)
        # goal_pose=np.array([0, 0, 0.26, 0])
        goal_msg = PoseStamped()
        goal_msg.header.frame_id = "map"
        goal_msg.header.stamp = rospy.Time.now()
        # if self.counter_episode == 1:
        obs = self._get_observation()
        self.gazebo.unpause()
        self.uav.ctrler.takeoff()

        if self.render_mode == "human":
            self.render()

        return obs, self._get_info()

    def render(self):

        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[box2d]`"
            )
        
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((VIEWPORT_W, VIEWPORT_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((VIEWPORT_W, VIEWPORT_H))

        pygame.transform.scale(self.surf, (SCALE, SCALE))
        pygame.draw.rect(self.surf, (255, 255, 255), self.surf.get_rect())

        self.surf = pygame.transform.flip(self.surf, False, True)

        if self.render_mode == "human":
            assert self.screen is not None
            self.screen.blit(self.surf, (0, 0))
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.surf)), axes=(1, 0, 2)
            )

    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
############################################################################################################

    def _get_reward(self, action, step):
        terminated = False
        truncated = False
        reward = 0.0
        max_episode_steps = 1024
        reward_max = 100
        success = False
        goal_z = self.wamv.gazebo_pose[2]+2 # 2m above wamv's bottom
        goal_z_tol = 0.2 # 20cm

        r = np.linalg.norm(self.uav.gazebo_pose[:1] - self.wamv.gazebo_pose[:1])
        h = self.uav.gazebo_pose[2] - goal_z

        u = lambda t: np.where(t >= 0, 1, 0)
        curve = lambda x: 2 / (1 + math.exp(x))

        hight_score = -reward_max*curve(abs(h+goal_z))*(1-u(h+goal_z_tol))+reward_max*curve(h)*u(h-r*2)
        #------------- under platform --------------------------------------- above platform
        dist_rew = hight_score*curve(r*(1-curve(h*0.5)))
        vz_punish = -10*self.uav_act[2]*u(h+goal_z_tol) + 20*self.uav_act[2]*(1-u(h+goal_z_tol))
        radius_bonus = 8*(self.prev_r - r)*u(h+goal_z_tol)
        
        reward += (
            + dist_rew
            + vz_punish
            + radius_bonus
        )

        self.prev_r = r
        self.prev_h = h

        if abs(self.uav.gazebo_pose[2] - goal_z) <= goal_z_tol and abs(r) <= 0.3:
            self.land_cnt += 1    
            reward = self.land_cnt*reward_max
            if self.land_cnt > 4:
                terminated = True
                reward = 10*reward_max
                success = True
        else:
            self.land_cnt = 0
        if self.counter_step >= max_episode_steps:
            terminated = True
                # reward = -100
        if self.uav.gazebo_pose[2] <= 0.0:# touch water
            terminated = True
            reward = -10*reward_max

        reward /= 1000
        return reward, terminated, truncated
    
    def _get_observation(self):
        # self.uwb_hist.push(self.uav.uwb.get_range())
        if self.uav.gazebo_pose is not None:
            self.uav_pose_hist.push(self.uav.gazebo_pose)
        if self.uav_act is not None:
            self.uav_act_hist.push(self.uav_act)
        # return {'uwb_ranges': self.uwb_hist.get(), 'uav_action': self.uav_act_hist.get()}
        return {'uav_action': self.uav_act_hist.get(), 'uav_gazebo_pose': self.uav_pose_hist.get()}

    def _move(self, x, y, z, yaw):
        self.uav.ctrler.move(x, y, z, yaw)

    def _get_info(self):
        return {}
    
    def _get_initial_state(self, name):
        state_msg = ModelState()
        state_msg.model_name = name
        x_rand = y_rand = 0 
        if name == 'wamv':
            state_msg.pose.position.z = 0
        elif name == 'sjtu_drone':
            self.uav.ctrler.takeoff()
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

