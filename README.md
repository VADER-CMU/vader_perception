# VADER Pose Estimation

## Attribution

This repository belongs to Carnegie Mellon University, Masters of Science - Robotic Systems Development (MRSD) Team E - VADER

Maintainer: Kshitij

Team Members: Tom Gao, Abhishek Mathur, Rohit Satishkumar, Kshitij Madhav Bhat, Keerthi GV 

First Revision: February 2025

## Introduction and Overview

This repository holds our pose estimation/perception stack, in charge of publishing coarse and fine pose estimates of peppers in the view of the gripper palm D405 Realsense camera.

## Installation & Usage
In your catkin workspace cd into `src`. Run the following commands to pull the latest version of `vader_perception` and other required packages.
```bash
git clone --depth 1 https://github.com/VADER-CMU/vader_perception.git
git clone https://github.com/VADER-CMU/vader_msgs.git
git clone https://github.com/VADER-CMU/realsense-ros.git
```
In the docker container, install `gdown`. (This will be added to vader_docker later)
```bash
pip install gdown
```

Catkin make in the `catkin_ws`
```bash
catkin_make
source devel/setup.bash
```

Launch the pose estimation node for the only the gripper camera
```bash
roslaunch vader_perception gripper_cam_pose_estimation.launch
```

Launch the pose estimation node for both cameras
```bash
roslaunch vader_perception dual_cam_pose_estimation.launch
```
Note: One of the cameras may not start. Roslaunch again in that case.