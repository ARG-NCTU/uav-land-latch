import os
from typing import Optional

import numpy as np
import yaml

import gymnasium as gym
from gymnasium import spaces

EPISODE_LEN = 128

class RectangleGrid:
    def __init__(self, vertices):
        self.vertices = vertices
        self.bounds = self.calculate_bounds()

    def calculate_bounds(self):
        x_coords = self.vertices[:, 0]
        y_coords = self.vertices[:, 1]
        return np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)

    def determine_region(self, points):
        px, py = points[:, 0], points[:, 1]
        min_x, max_x, min_y, max_y = self.bounds

        conditions = np.array(
            [
                (px > max_x) & (py > max_y),
                (min_x <= px) & (px <= max_x) & (py > max_y),
                (px < min_x) & (py > max_y),
                (px < min_x) & (min_y <= py) & (py <= max_y),
                (px < min_x) & (py < min_y),
                (min_x <= px) & (px <= max_x) & (py < min_y),
                (px > max_x) & (py < min_y),
                (px > max_x) & (min_y <= py) & (py <= max_y),
                (min_x <= px) & (px <= max_x) & (min_y <= py) & (py <= max_y),
            ]
        )

        return np.argmax(conditions, axis=0) + 1

    def regions_to_nlos_masks(self, regions):
        masks = np.array(
            [
                #    0  1  2  3  4  5
                [0, 0, 1, 0, 1, 0],  # 0
                [0, 1, 1, 0, 1, 0],  # 1
                [0, 1, 0, 0, 1, 0],  # 2
                [0, 1, 0, 1, 1, 1],  # 3
                [0, 0, 0, 1, 0, 1],  # 4
                [1, 0, 0, 1, 0, 1],  # 5
                [1, 0, 0, 0, 0, 1],  # 6
                [1, 0, 1, 0, 1, 1],  # 7
                [0, 0, 0, 0, 0, 0],  # 8
            ]
        )  # 0 for LOS, 1 for NLOS
        return masks[regions - 1]

class DroneRangeLandV3(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render_mode=None, seed=0):
        self.seed = seed
        self.max_speed = 1.0
        self.max_angular_speed = 1.0
        self.dt = 0.1

        action_high = np.array(
            [self.max_speed, self.max_speed, self.max_speed, self.max_angular_speed], dtype=np.float64
        )  # vx, vy, vz, vyaw
        self.action_space = spaces.Box(low=-action_high, high=action_high, dtype=np.float64, seed=seed)

        observation_low = np.full((12 + 12,), 0.0, dtype=np.float64)
        observation_high = np.full((12 + 12,), 1.0, dtype=np.float64)
        self.observation_space = spaces.Box(low=observation_low, high=observation_high, dtype=np.float64)

        self.drone_pose = np.zeros(4, dtype=np.float64)  # x, y, z, yaw
        self.drone_signature = np.zeros(6, dtype=np.float64)

        self.goal_pose = np.zeros(4, dtype=np.float64)  # x, y, z, yaw
        self.goal_signature = np.zeros(6, dtype=np.float64)

        self.last_dist = 0.0

        self.obstable = {"x": [-1.9, 2.8], "y": [-1.5, 1.5], "z": [-np.inf, 0.5]}

        self.uwb_anchor_pos = self.load_uwb_pos_from_yaml(
            os.path.expanduser("~/robotx-2022/catkin_ws/src/pozyx_ros/config/wamv/wamv.yaml")
        )
        self.num_anchors = self.uwb_anchor_pos.shape[0]

        self.uwb_tag_pos = self.load_uwb_pos_from_yaml(
            os.path.expanduser("~/robotx-2022/catkin_ws/src/pozyx_ros/config/wamv/drone.yaml")
        )
        self.num_tags = self.uwb_tag_pos.shape[0]
        self.rectangles = RectangleGrid(self.uwb_anchor_pos[:4, :2])
        print(f"Environment initialized: {seed}")
        np.set_printoptions(precision=3, suppress=True)

    def load_uwb_pos_from_yaml(self, yaml_file_path):
        with open(yaml_file_path, "r") as file:
            data = yaml.safe_load(file)
        positions = np.array([[item["x"], item["y"], item["z"]] for item in data.values()]) / 1000.0
        return positions

    def step(self, action):
        dx, dy, dz, dyaw = action * self.dt
        yaw = self.drone_pose[3]
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # Update position and yaw
        self.drone_pose[0] += cos_yaw * dx - sin_yaw * dy
        self.drone_pose[1] += sin_yaw * dx + cos_yaw * dy
        self.drone_pose[2] += dz
        self.drone_pose[3] = self.angle_normalize(yaw + dyaw)

        observation = np.concatenate([self.goal_signature, self.pose_to_signature(self.drone_pose)])
        terminated = self.is_inside_obstacle(*self.drone_pose[0:3])

        dist_xyz = np.linalg.norm(self.drone_pose[0:3] - self.goal_pose[0:3])
        dist_yaw = np.abs(self.angle_normalize(self.drone_pose[3] - self.goal_pose[3]))
        dist = dist_xyz + 0.5 * dist_yaw

        reward = 2.0 * (self.last_dist - dist) - dist / EPISODE_LEN + min(10.0, 1.0 / dist) / EPISODE_LEN
        reward -= 2.0 if terminated else 0.0

        reward += (
            min(1.0 / (self.drone_pose[2] - self.obstable["z"][1]), 5.0)
            if self.obstable["x"][0] < self.drone_pose[0] < self.obstable["x"][1]
            and self.obstable["y"][0] < self.drone_pose[1] < self.obstable["y"][1]
            and self.obstable["z"][1] < self.drone_pose[2] <= self.obstable["z"][1] + 2.0
            else 0.0
        ) / EPISODE_LEN
        # reward += 1.0 if self.is_near_goal() else 0.0

        self.last_dist = dist

        return observation, reward, terminated, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.seed == 0:
            print(f"Goal: {self.goal_pose}, Final Drone: {self.drone_pose}")
            print(
                f"Final Error: {np.linalg.norm(self.goal_pose[0:3] - self.drone_pose[0:3]) + 0.5 * np.abs(self.angle_normalize(self.drone_pose[3] - self.goal_pose[3]))}"
            )
        self.respawn_range = [(-0.2, 0.2), (-0.1, 0.1), (0.5, 1.5), (-np.pi, np.pi)]

        self.goal_pose = np.array([self.np_random.uniform(low, high) for low, high in self.respawn_range])
        self.respawn_range = [(-5, 5), (-5, 5), (-1, 5), (-np.pi, np.pi)]

        while True:
            self.drone_pose = np.array([self.np_random.uniform(low, high) for low, high in self.respawn_range])
            if not self.is_inside_obstacle(*self.drone_pose[0:3]):
                break

        self.goal_signature = self.pose_to_signature(self.goal_pose)
        observation = np.concatenate([self.goal_signature, self.pose_to_signature(self.drone_pose)])
        if self.seed == 0:
            print(f"Goal: {self.goal_pose}, Drone: {self.drone_pose}")
            print(
                f"Error: {np.linalg.norm(self.goal_pose[0:3] - self.drone_pose[0:3]) + 0.5 * np.abs(self.angle_normalize(self.drone_pose[3] - self.goal_pose[3]))}"
            )
        self.last_dist = np.linalg.norm(self.goal_pose[0:3] - self.drone_pose[0:3]) + 0.5 * np.abs(
            self.angle_normalize(self.drone_pose[3] - self.goal_pose[3])
        )
        return observation, {}

    def pose_to_signature(self, pose: np.ndarray) -> np.ndarray:
        uwb_tag_pos = self.body_pose_to_uwb_tag_pos(pose)
        regions = self.rectangles.determine_region(uwb_tag_pos[:, :2])
        nlos_masks = self.rectangles.regions_to_nlos_masks(regions).flatten()

        distances = np.linalg.norm(uwb_tag_pos[:, None, :] - self.uwb_anchor_pos, axis=2).flatten()
        distances[nlos_masks == 1] = 1.48 * distances[nlos_masks == 1] + 0.289
        signature = distances / 50.0
        return signature

    def body_pose_to_uwb_tag_pos(self, pose: np.ndarray) -> np.ndarray:
        x, y, z, yaw = pose
        cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)

        # Precompute rotation matrix components
        rotation_matrix = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

        # Transform UWB tag positions in the drone frame to the world frame
        uwb_tag_pos_2d = self.uwb_tag_pos[:, :2]
        tag_pos_transformed_2d = np.dot(uwb_tag_pos_2d, rotation_matrix.T) + np.array([x, y])

        # Create the final transformed positions with the z coordinate
        tag_pos_transformed = np.empty_like(self.uwb_tag_pos)
        tag_pos_transformed[:, :2] = tag_pos_transformed_2d
        tag_pos_transformed[:, 2] = z

        return tag_pos_transformed

    def is_inside_obstacle(self, x, y, z):
        return (
            self.obstable["x"][0] <= x <= self.obstable["x"][1]
            and self.obstable["y"][0] <= y <= self.obstable["y"][1]
            and self.obstable["z"][0] <= z <= self.obstable["z"][1]
        )

    def is_near_goal(self):
        return (
            np.linalg.norm(self.goal_pose[0:3] - self.drone_pose[0:3]) < 0.2
            and np.abs(self.angle_normalize(self.drone_pose[3] - self.goal_pose[3])) < 0.25
        )

    @staticmethod
    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi
