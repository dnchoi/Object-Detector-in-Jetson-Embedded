#!/bin/bash
for i in "$@"; do
  case $i in
    -i=*|--ip=*)
      ROS_IP="${i#*=}"
      shift # past argument=value
      ;;
    -m=*|--master=*)
      ROS_MASTER="${i#*=}"
      shift # past argument=value
      ;;
    -use=*|--use_model=*)
      USE_MODEL="${i#*=}"
      shift # past argument=value
      ;;
    -n=*|--node=*)
      NODE="${i#*=}"
      shift # past argument=value
      ;;
    -w=*|--width=*)
      WIDTH="${i#*=}"
      shift # past argument=value
      ;;
    -h=*|--height=*)
      HEIGHT="${i#*=}"
      shift # past argument=value
      ;;
    -c=*|--camera=*)
      CAMERA="${i#*=}"
      shift # past argument=value
      ;;
  esac
done
echo "ROS_IP  = ${ROS_IP}"
echo "ROS_MASTER     = ${ROS_MASTER}"
echo "USE_MODEL    = ${USE_MODEL}"
echo "NODE         = ${NODE}"
echo "WIDTH         = ${WIDTH}"
echo "HEIGHT         = ${HEIGHT}"
echo "CAMERA         = ${CAMERA}"

export ROS_IP=$ROS_IP
export ROS_MASTER_URI=http://$ROS_MASTER:11311
source devel/setup.bash; rosrun edge app_edge.py --use_model $USE_MODEL --node $NODE --app  --width $WIDTH --height $HEIGHT --rtsp $CAMERA
