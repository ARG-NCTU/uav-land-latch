# Train Drone RL using ACME PPO

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

## Run the training

Run ubuntu-20.04 docker

```
source Docker/ubuntu-20.04/run.bash
```

Start the training
```
python3 ./rl/jax/ppo.py
```

Start the tensorboard
```
tensorboard --logdir ~/acme
```
