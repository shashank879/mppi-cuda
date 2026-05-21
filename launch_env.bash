#!/bin/bash

# Automatically determine the server name based on the hostname
detect_server() {
  HOSTNAME=$(hostname)

  case "$HOSTNAME" in
    *"aching"*)
      SERVER_NAME="aching"
      ;;
    *"parkin"*)
      SERVER_NAME="parkin"
      ;;
    *"gpumachine22"*)
      SERVER_NAME="azure"
      ;;
    *"jade"*)
      SERVER_NAME="jade"
      ;;
    *"Shashank-UoB-Linux"*)
      SERVER_NAME="UoB"
      ;;
    *"angua"*)
      SERVER_NAME="angua"
      ;;
    *"shine"*)
      SERVER_NAME="shine"
      ;;
    *"dibbler"*)
      SERVER_NAME="dibbler"
      ;;
    *)
      echo "Unknown server based on hostname: $HOSTNAME"
      exit 1
      ;;
  esac
}

# Default server configurations
configure_server() {
  case "$SERVER_NAME" in
    "aching")
      GPU=""
      PORTS=("15900:5900")
      RESOLUTION="1600x900"
      VOLUMES=("/mnt/faster0/ss3966/:/home/ss3966/")
      HOME_DIR="/home/ss3966/"
      IMAGE="ss3966/phd"
      CMD_TYPE="hare"
      ;;
    "parkin")
      GPU=""
      PORTS=("15900:5900")
      RESOLUTION="1600x900"
      VOLUMES=("/mnt/faster0/ss3966/:/home/ss3966/")
      HOME_DIR="/home/ss3966/"
      IMAGE="ss3966/test"
      CMD_TYPE="hare"
      ;;
    "angua")
      GPU=""
      RESOLUTION="1600x900"
      VOLUMES=("/mnt/faster0/ss3966/:/home/ss3966/")
      HOME_DIR="/home/ss3966/"
      IMAGE="shashank879/gpa:v2.3"
      CMD_TYPE="hare"
      ;;
    "shine")
      GPU=""
      RESOLUTION="1600x900"
      VOLUMES=("/mnt/faster2/ss3966/:/home/ss3966/")
      HOME_DIR="/home/ss3966/"
      IMAGE="shashank879/gpa:v2.3"
      CMD_TYPE="hare"
      ;;
    "dibbler")
      GPU=""
      RESOLUTION="1600x900"
      VOLUMES=("/mnt/fast0/ss3966/:/home/ss3966/")
      HOME_DIR="/home/ss3966/"
      IMAGE="shashank879/gpa:v3.1"
      CMD_TYPE="hare"
      ;;
    "azure")
      GPU=""
      RESOLUTION="1024x768"
      VOLUMES=("/mnt/faster0/ss3966/:/home/ss3966/")
      IMAGE="shashank879/gpa:v2.3"
      HOME_DIR="/home/ss3966/"
      CMD_TYPE="docker"
      ;;
    "jade")
      GPU="gpu:1"
      IMAGE="../gpa.img"
      CMD_TYPE="srun"
      ;;
    "UoB")
      GPU="0"
      RESOLUTION="1024x768"
      VOLUMES=("/mnt/hd/ss3966/:/home/ss3966/")
      IMAGE="shashank879/gpa:v2.3"
      HOME_DIR="/home/ss3966/"
      CMD_TYPE="docker"
      ;;
    *)
      echo "Unknown server: $SERVER_NAME"
      exit 1
      ;;
  esac
}

# Function to parse command-line arguments
parse_overrides() {
  while [[ $# -gt 0 ]]; do
    key="$1"

    case $key in
      --gpu)
        GPU="$2"
        shift # past argument
        shift # past value
        ;;
      --port)
        PORTS+=("$2")
        shift
        shift
        ;;
      --resolution)
        RESOLUTION="$2"
        shift
        shift
        ;;
      --image)
        IMAGE="$2"
        shift
        shift
        ;;
      --volume)
        VOLUMES+=("$2")
        shift
        shift
        ;;
      --home)
        HOME_DIR="$2"
        shift
        shift
        ;;
      *)
        echo "Unknown option $key"
        exit 1
        ;;
    esac
  done
}

# Detect the server automatically
detect_server

# Configure the server based on the detected name
configure_server

# Parse additional command-line arguments for overrides
parse_overrides "$@"

# Execute the command based on the configuration type
if [ "$CMD_TYPE" = "docker" ] || [ "$CMD_TYPE" = "hare" ]; then
  CMD="$CMD_TYPE run -it --rm -u $(id -u):$(id -g) --gpus \"device=$GPU\" -e VNC_PASSWORD=12345678 -e RESOLUTION=$RESOLUTION"

  # Add HOME environment variable if set
  if [ -n "$HOME_DIR" ]; then
    CMD="$CMD -e HOME=$HOME_DIR"
  fi

  # Add volumes
  for VOLUME in "${VOLUMES[@]}"; do
    CMD="$CMD -v $VOLUME"
  done

  # Add port mappings if defined
  for PORT in "${PORTS[@]}"; do
    CMD="$CMD -p $PORT"
  done

  CMD="$CMD $IMAGE"

elif [ "$CMD_TYPE" = "srun" ]; then
  CMD="srun -I --pty --gres $GPU -p big singinteractive $IMAGE"
fi

# Execute the final command
echo "Running: $CMD"
eval $CMD
