#! /usr/bin/env python3
import os
from datetime import datetime

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    ProgressBarCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_uav.feature_extractor import UAVMultiExtractor
import gymnasium as gym
import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.env_util import make_vec_env
from sb3_uav.callback import SaveSameModelCB
from stable_baselines3 import PPO, A2C

from datetime import date

max_steps = 2048
vec_env = make_vec_env("gymnasium_uav_classic:uav-lander-v5", n_envs=1)

policy_kwargs = dict(
    features_extractor_class=UAVMultiExtractor,
)

today = date.today()
checkpoint_callback = SaveSameModelCB(
    save_freq=100000,
    save_path="./logs/",
    name_prefix="ppo_uavlander_cheat" + str(today),
    save_replay_buffer=True,
    save_vecnormalize=True,
    overwrite=True,
    verbose=0,
    hist_checkpoint_len=5,
    checkpoint_dir_name="ppo_uavlander_checkpoints"
)

model = PPO.load("logs/uavlander_cheat_1500000_steps")
model.set_env(vec_env)
model.learn(total_timesteps=20_000_000, tb_log_name='tb_ppo', callback=checkpoint_callback)
model.save("ppo_uavlander")