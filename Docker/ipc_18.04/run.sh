#!/usr/bin/env bash

ARGS=("$@")

REPOSITORY="argnctu/robotx2022"
TAG="ipc-18.04"

IMG="${REPOSITORY}:${TAG}"

USER_NAME="argrobotx"
REPO_NAME="uav-land-latch"
CONTAINER_NAME="argrobotx-ipc-18.04"

# Get running container ID
RUNNING_CONTAINER_ID=$(docker ps -qf "name=^${CONTAINER_NAME}$")
# Get *any* container ID (running or exited)
EXISTING_CONTAINER_ID=$(docker ps -aqf "name=^${CONTAINER_NAME}$")

if [ -n "$RUNNING_CONTAINER_ID" ]; then
  echo "Attaching to running container $RUNNING_CONTAINER_ID"
  xhost +
  docker exec --privileged -e DISPLAY=${DISPLAY} -e LINES="$(tput lines)" -it "$RUNNING_CONTAINER_ID" bash
  xhost -
  return
elif [ -n "$EXISTING_CONTAINER_ID" ]; then
  echo "Removing exited container $EXISTING_CONTAINER_ID"
  docker rm "$EXISTING_CONTAINER_ID"
fi

# Setup xauth for X11
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]; then
  xauth_list=$(xauth nlist $DISPLAY)
  xauth_list=$(sed -e 's/^..../ffff/' <<<"$xauth_list")
  if [ ! -z "$xauth_list" ]; then
    echo "$xauth_list" | xauth -f $XAUTH nmerge -
  else
    touch $XAUTH
  fi
  chmod a+r $XAUTH
fi

if [ ! -f $XAUTH ]; then
  echo "[$XAUTH] was not properly created. Exiting..."
  exit 1
fi

# Now run container
docker run \
  -it \
  --runtime=nvidia \
  -e DISPLAY \
  -e XAUTHORITY=$XAUTH \
  -e HOME=/home/${USER_NAME} \
  -e OPENAI_API_KEY=$OPENAI_API_KEY \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -v "$XAUTH:$XAUTH" \
  -v "/home/${USER}/${REPO_NAME}:/home/${USER_NAME}/${REPO_NAME}" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "/etc/localtime:/etc/localtime:ro" \
  -v "/dev:/dev" \
  -v "/var/run/docker.sock:/var/run/docker.sock" \
  --user "root:root" \
  --workdir "/home/${USER_NAME}/${REPO_NAME}" \
  --name "${CONTAINER_NAME}" \
  --network host \
  --privileged \
  --security-opt seccomp=unconfined \
  "${IMG}" \
  bash