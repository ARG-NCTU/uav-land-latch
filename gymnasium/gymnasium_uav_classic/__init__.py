# here's where you register your envs
from gymnasium.envs.registration import make, register, registry, spec

register(
    id="uav-lander-v5",
    entry_point="gymnasium_uav_classic.envs:UavLanderV5",
    max_episode_steps=1024,
)