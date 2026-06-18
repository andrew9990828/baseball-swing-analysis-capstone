"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: pose_pipeline.py
Description:
    Pipeline for extracting pose landmarks from an entire video.
    This module opens a video, runs pose estimation frame-by-frame,
    saves raw landmark arrays, and writes debug images.

Last Updated: 4/27/26

Notes:
    This file coordinates video reading, pose estimation, debug image saving,
    and NumPy output storage for Module 2.
"""

from pathlib import Path

import cv2 as cv
import numpy as np

from src.pose.pose_estimator import MediaPipePoseEstimator


"""
Module 2 mental model:

1. Open video with OpenCV.
2. Read one frame at a time.
3. Send each frame to MediaPipePoseEstimator.
4. Store each frame's (33, 4) landmark array.
5. Stack all frames into one (num_frames, 33, 4) array.
6. Save landmarks, frame_indices, and pose_detected to .npz.
7. Save visual debug frames to confirm pose detection worked.
"""

def extract_pose_from_video(
    video_path: str,
    output_path: str,
    debug_dir: str,
    max_frames: int | None = None,
    debug_every_n_frames: int = 10,
) -> None:
    """
    Extract pose landmarks from a video and save standardized outputs.

    Args:
        video_path: Path to the input video file.
        output_path: Path where the .npz landmark data should be saved.
        debug_dir: Directory where annotated debug frames should be saved.
        max_frames: Optional cap on number of frames to process.
        debug_every_n_frames: Save one annotated debug frame every N frames.
    """

    # TODO 1: Convert strings into Path objects.
    video_pth = Path(video_path)
    output_pth = Path(output_path)
    debug_pth = Path(debug_dir)

    # TODO 2: Create output folders.
    output_pth.parent.mkdir(parents=True, exist_ok=True)
    debug_pth.mkdir(parents=True, exist_ok=True)

    # TODO 3: Open the video with cv.VideoCapture.
    capture = cv.VideoCapture(str(video_pth))

    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_pth}")

    # TODO 4: Create the MediaPipePoseEstimator.
    estimator = MediaPipePoseEstimator()

    all_landmarks = []
    frame_indices = []
    detection_flags = []

    frame_index = 0

    # TODO 5: Loop through frames.
    while True:
        success, frame = capture.read()

        if not success:
            break

        if max_frames is not None and frame_index >= max_frames:
            break

        result = estimator.estimate_frame(frame, frame_index)

        # TODO 6: Save landmarks, frame indices, and detection flags.
        all_landmarks.append(result.landmarks)
        frame_indices.append(result.frame_index)
        detection_flags.append(result.pose_detected)

        # Shows the pose every 10 frames basically so we can debug
        if frame_index % debug_every_n_frames == 0:
            annotated_frame = estimator.draw_landmarks(frame)
            debug_path = debug_pth / f"frame_{frame_index:04d}_pose.jpg"
            cv.imwrite(str(debug_path), annotated_frame)

        frame_index += 1

    # TODO 7: Release video and estimator resources.
    capture.release()
    estimator.close()

    if len(all_landmarks) == 0:
        raise RuntimeError(f"No frames were processed from video: {video_pth}")

    landmarks_array = np.stack(all_landmarks, axis=0)
    frame_indices_array = np.array(frame_indices, dtype=np.int32)
    detection_flags_array = np.array(detection_flags, dtype=bool)

    np.savez(
        output_pth,
        landmarks=landmarks_array,
        frame_indices=frame_indices_array,
        pose_detected=detection_flags_array,
    )

    detected_count = int(np.sum(detection_flags_array))
    total_count = len(detection_flags_array)

    print("Pose extraction complete.")
    print(f"Frames processed: {total_count}")
    print(f"Frames with pose detected: {detected_count}")
    print(f"Saved pose data to: {output_pth}")
    print(f"Saved debug frames to: {debug_pth}")