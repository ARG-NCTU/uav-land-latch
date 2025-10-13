#! /usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

source "$SCRIPT_DIR/01_speedup_kill_gazebo.bash"
source "$SCRIPT_DIR/02_add_env_to_bashrc.sh"
source "$SCRIPT_DIR/03_rosconsole_format.sh"
source "$SCRIPT_DIR/04_fix_python3_path.sh"

source ~/.bashrc
