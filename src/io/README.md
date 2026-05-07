# IO Module

This folder contains video loading utilities.

## Purpose

Load raw swing video files and convert them into frame data that later modules can process.

## Files

- `video_loader.py`
  - loads video metadata
  - extracts frames
  - attaches timestamps
  - supports debug frame saving

## Engineering Decisions

- OpenCV is used because it is simple, reliable, and works well for frame extraction.
- Frames are stored with frame index and timestamp so later modules can connect movement back to time.
- Debug frames are saved to confirm the video loaded correctly and to inspect video quality.

## Status

Module 1 is complete for v1.