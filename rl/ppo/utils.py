import os

from stable_baselines3.common.monitor import Monitor

import gymnasium as gym


def linear_schedule(initial_value, final_value=0.0, start_progress=0.0, end_progress=1.0):
    if isinstance(initial_value, str):
        initial_value = float(initial_value)
    if isinstance(final_value, str):
        final_value = float(final_value)

    assert 0.0 <= start_progress < end_progress <= 1.0, "Progress values must be between 0 and 1, with start < end."

    def scheduler(progress):
        progress = 1.0 - progress
        return_value = None
        if progress < start_progress:
            return_value = initial_value
        elif progress > end_progress:
            return_value = final_value
        else:
            # Scale the progress to be within start_progress and end_progress
            scaled_progress = (progress - start_progress) / (end_progress - start_progress)
            return_value = initial_value + scaled_progress * (final_value - initial_value)
        return return_value

    return scheduler


def make_env(seed=0, id="gymnasium_vrx:drone-range-yz-v1", monitor_dir=None):
    def _init():
        # env = gym.make("gymnasium_vrx:drone-range-yz-v1", seed=seed)
        # env = gym.make("gymnasium_vrx:drone-range-yz-simple-v0", seed=seed)
        # env = gym.make("gymnasium_vrx:drone-range-simple-v0", seed=seed)
        # env = gym.make("gymnasium_vrx:drone-range-land-v0", seed=seed)
        env = gym.make(id, seed=seed)
        if monitor_dir is not None:
            os.makedirs(monitor_dir, exist_ok=True)
        env = Monitor(env, monitor_dir)
        return env

    return _init
