# uav-land-latch


## Clone repo

```bash
git clone --recursive git@github.com:wilbur1240/uav-land-latch.git
```

## Update repo and submodules

```bash
git pull
git submodule sync --recursive
git submodule update --init --recursive
```

## Set up Docker
The requried dependencies are installed, you only need a PC with GPU, and make sure it install docker already.

## How to run

1. Docker Run

    Run this script to pull docker image to your workstation.

    ```
    source ipc_run.sh
    ```

2. Docker Join

    If want to enter same docker image, type below command.

    ```
    source ipc_join.sh
    ```

3. Catkin_make

    Execute the compile script at first time, then the other can ignore this step. 

    ```
    source build_all.sh
    ```

4. Setup environment

    Make sure run this command when the terminal enter docker. 

    ```
    source environment.sh
    ```

## robotx-2022 legacy UAV land latch

[UAV_Landing_Latching](docs/UAV_Landing_Latching)