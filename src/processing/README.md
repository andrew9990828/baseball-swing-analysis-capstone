# Processing Module

This folder contains landmark processing utilities for preparing pose data for swing analysis.

## Purpose

Convert raw pose landmark output into cleaner, normalized motion data.

## Files

- `pose_pipeline.py`
  - runs pose extraction across a full video
  - saves raw pose data and debug outputs

- `landmark_processor.py`
  - loads raw Module 2 pose data
  - extracts the `landmarks` array from `.npz`
  - smooths landmark trajectories
  - normalizes landmarks around hip center
  - saves cleaned landmark output

## Engineering Decisions

- Module 2 saves pose data as a `.npz` archive containing:
  - `landmarks`
  - `frame_indices`
  - `pose_detected`

- Module 3 specifically loads the `landmarks` array.

- Landmark shape:

```text
(num_frames, num_landmarks, 4)