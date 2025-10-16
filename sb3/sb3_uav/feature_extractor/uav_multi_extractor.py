from typing import Callable, Dict, List, Optional, Tuple, Type, Union
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class UAVMultiExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 1):
        super(UAVMultiExtractor, self).__init__(observation_space, features_dim)
        extractors = {}
        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'uwb_ranges':
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                )
                total_concat_size += subspace.shape[0]*subspace.shape[1] # related to uwb range number
            elif key == 'uav_action':
                extractors[key] = nn.Sequential(
                        nn.Flatten(),
                    )
                total_concat_size += subspace.shape[0]*subspace.shape[1] # relate to uav action space
            elif key == 'uav_gazebo_pose':
                extractors[key] = nn.Sequential(
                    nn.Flatten(),
                )
                total_concat_size += subspace.shape[0]*subspace.shape[1]
            else:
                # Run through nothing
                extractors[key] = nn.Sequential()
                total_concat_size += subspace.shape[0]
        
        self.extractors = nn.ModuleDict(extractors)
        self.mlp_network = nn.Sequential(
            nn.Linear(total_concat_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self._features_dim = 256

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        features = th.cat(encoded_tensor_list, dim=1)
        return self.mlp_network(features)