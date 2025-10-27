# VADER Pose Estimation

## Attribution

This repository belongs to Carnegie Mellon University, Masters of Science - Robotic Systems Development (MRSD) Team E - VADER

Team Members: Tom Gao, Abhishek Mathur, Rohit Satishkumar, Kshitij Bhat, Keerthi GV 

First Revision: February 2025

## Introduction and Overview

This repository holds our pose estimation/perception stack, in charge of publishing coarse and fine pose estimates of peppers in the view of the gripper palm D405 Realsense camera.

## Installation

Clone this repository into your workspace. In addition to this, make sure that the realsense-ros ROS package from https://github.com/rjwb1/realsense-ros is cloned into the src folder of your workspace separately.

## Usage

```bash
git clone https://github.com/VADER-CMU/vader_perception.git --branch svd-archive
cd vader_perception/
git submodule update --init --recursive
catkin_make
source devel/setup.bash
```

Use ROSLaunch with `coarse_pose_estimation.launch` to launch the pose estimation script.
