# Pose Module

This folder contains Module 2: Pose Extraction Interface.

## Purpose

Run pose estimation on video frames and convert the model output into a standard landmark format.

## Files

- `pose_estimator.py`
  - defines pose result containers
  - runs MediaPipe Pose on one frame
  - returns landmark data
  - can draw debug skeletons on frames

## Output Format

Each detected frame produces 33 MediaPipe landmarks.

Each landmark stores:

```text
x, y, z, visibility