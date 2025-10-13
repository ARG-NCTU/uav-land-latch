# Train Drone RL using Stable-baselines3 PPO

## Run the Gazebo simulation

Run ipc-18.04 docker
```
source ipc_run.sh
```

Source the environment
```
source environment.sh
```

Run the simulation
```
make drone_uwb_sjtu
```

## Before training

Prepare the training config

For example: [`./rl/ppo/config/ppo_uav_range_land.yaml`](/rl/ppo/config/ppo_uav_range_land.yaml)

```yaml
task_name: "uav-range-track-ppo"
env_id: "gymnasium_vrx:drone-range-track-v2"
env_script_path: "~/robotx-2022/gymnasium/gymnasium_vrx/envs/drone_range_track_v2.py"
n_envs: 2048
policy_kwargs:
  net_arch:
   - 256
   - 256
training_steps: 50.0e+9
ppo:
  learning_rate: 3.0e-4
  n_steps: 32
  batch_size: 65536
  n_epochs: 2048
  gamma: 0.99
  ent_coef: 0.002
  target_kl: 1.0
checkpoint:
  save_freq: 10000
```

## Run the training

Run ubuntu-20.04 docker

```
source Docker/ubuntu-20.04/run.bash
```

Install gymnasium-vrx

```
source install_gymnasium.sh
```

Start the training
```
python3 ./rl/ppo/ppo_run_uav.py
```

Start the tensorboard
```
tensorboard --logdir=rl/ppo
```
