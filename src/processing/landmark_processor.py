"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: landmark_processor.py
Description:
    Utilities for processing raw pose landmark arrays after pose estimation.
    This module smooths noisy landmark trajectories, normalizes landmark
    coordinates around a body reference point, and prepares motion data for
    downstream swing phase detection and feature extraction.

Last Updated: 4/30/26

Notes:
    Code implementation is being developed independently by Andrew Bieber.
    High-level scaffolding, planning, and pseudocode were discussed in collaboration.


Module 3: Landmark Processing

Purpose:
    Convert raw pose landmark arrays into cleaner, normalized motion data
    that can be used for event detection and feature extraction.

This module takes the saved output from the pose extraction stage and prepares
it for real swing analysis.
"""

from pathlib import Path
import numpy as np


def load_landmarks(landmark_path: str | Path) -> np.ndarray:
    """
    Load saved pose landmarks from a NumPy file.

    Supports:
        .npy = direct NumPy array
        .npz = compressed NumPy archive

    For Module 2 pose output, the .npz file contains:
        landmarks
        frame_indices
        pose_detected

    Module 3 uses:
        landmarks
    """
    landmark_path = Path(landmark_path)

    if not landmark_path.exists():
        raise FileNotFoundError(f"Landmark file not found: {landmark_path}")

    loaded = np.load(landmark_path)

    if landmark_path.suffix == ".npz":
        if "landmarks" not in loaded.files:
            raise KeyError(
                f"'landmarks' not found in {landmark_path}. "
                f"Available keys: {loaded.files}"
            )

        landmarks = loaded["landmarks"]
    else:
        landmarks = loaded

    return landmarks


def inspect_landmarks(landmarks: np.ndarray) -> None:
    """
    Print basic information about the landmark array.
    """
    print("Landmark array shape:", landmarks.shape)
    print("Number of frames:", landmarks.shape[0])
    print("Number of landmarks:", landmarks.shape[1])
    print("Values per landmark:", landmarks.shape[2])
    print("Min value:", np.nanmin(landmarks))
    print("Max value:", np.nanmax(landmarks))


def extract_landmark_trajectory(
    landmarks: np.ndarray,
    landmark_index: int,
    values: str = "xy" ) -> np.ndarray:
    """
    Extract one landmark across all frames.

    Args:
        landmarks:
            Full landmark array with shape:
            (num_frames, num_landmarks, values)

        landmark_index:
            MediaPipe landmark index to extract.

        values:
            "xy" returns x and y coordinates.
            "xyz" returns x, y, and z coordinates.
            "visibility" returns visibility only.

    Returns:
        A NumPy array containing that landmark's movement over time.
    """
    if values == "xy":
        return landmarks[:, landmark_index, 0:2]

    if values == "xyz":
        return landmarks[:, landmark_index, 0:3]

    if values == "visibility":
        return landmarks[:, landmark_index, 3]

    raise ValueError(f"Unsupported values option: {values}")


# Make sure we test out what smoothing window size actually is most optimal
# Too little we can have jitter, too large the data is practically useless
# Start a default of 5, then test 3, then test 7. Pick what seems most ideal.
def smooth_trajectory(trajectory: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooth a landmark trajectory using a simple moving average.

    This helps reduce frame-to-frame pose jitter.
    """
    if window_size <= 1:
        return trajectory

    smoothed = np.zeros_like(trajectory)

    half_window = window_size // 2

    for i in range(len(trajectory)):
        start = max(0, i - half_window)
        end = min(len(trajectory), i + half_window + 1)

        smoothed[i] = np.mean(trajectory[start:end], axis=0)

    return smoothed


# Make sure we test out what smoothing window size actually is most optimal
# Too little we can have jitter, too large the data is practically useless
# Start a default of 3, then test 5, then test 7. Pick what seems most ideal.
def smooth_all_landmarks(landmarks: np.ndarray, window_size: int = 3) -> np.ndarray:
    """
    Apply smoothing to x, y, z values for every landmark across all frames.

    Visibility is preserved without smoothing.
    """
    smoothed = landmarks.copy()

    num_landmarks = landmarks.shape[1]

    for landmark_index in range(num_landmarks):
        trajectory_xyz = landmarks[:, landmark_index, 0:3]
        smoothed[:, landmark_index, 0:3] = smooth_trajectory(
            trajectory_xyz,
            window_size=window_size
        )

    return smoothed


def get_body_center(landmarks: np.ndarray) -> np.ndarray:
    """
    Calculate a body reference point using the midpoint between left and right hips.

    MediaPipe Pose indices:
        23 = left hip
        24 = right hip

    Returns:
        Body center trajectory with shape:
        (num_frames, 3)
    """
    left_hip = landmarks[:, 23, 0:3]
    right_hip = landmarks[:, 24, 0:3]

    body_center = (left_hip + right_hip) / 2.0
    return body_center


def normalize_to_body_center(landmarks: np.ndarray) -> np.ndarray:
    """
    Normalize landmark coordinates relative to the hip center.

    This makes movement more body-relative instead of screen-position-relative.
    """
    normalized = landmarks.copy()

    body_center = get_body_center(landmarks)

    normalized[:, :, 0:3] = landmarks[:, :, 0:3] - body_center[:, np.newaxis, :]

    return normalized


# smooth_trajectory keeps a default window of 5 for standalone testing.
# The full landmark-processing pipeline uses a smaller window of 3 by default
# because baseball swing motion is fast and sharp timing changes matter.
def process_landmarks(
    input_path: str | Path,
    output_path: str | Path,
    smoothing_window: int = 3 ) -> np.ndarray:
    """
    Full Module 3 processing pipeline.

    Steps:
        1. Load raw landmarks
        2. Smooth landmark movement
        3. Normalize coordinates to body center
        4. Save processed landmarks
    """
    landmarks = load_landmarks(input_path)

    print("Raw landmarks:")
    inspect_landmarks(landmarks)

    smoothed = smooth_all_landmarks(
        landmarks,
        window_size=smoothing_window
    )

    normalized = normalize_to_body_center(smoothed)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_path, normalized)

    print(f"Processed landmarks saved to: {output_path}")
    print("Processed landmarks:")
    inspect_landmarks(normalized)

    return normalized