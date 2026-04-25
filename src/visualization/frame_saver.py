"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: frame_saver.py
Description:
    Utility functions for saving extracted video frames to disk and
    creating simple debug outputs for pipeline verification.

Last Updated: 4/25/26

Notes:
    Code implementation is being developed independently by Andrew Bieber.
    High-level scaffolding, planning, and pseudocode were discussed in collaboration.
"""

from pathlib import Path
import cv2 as cv

def save_frame_image(frame, output_path: str) -> bool:

    if frame is None:
        raise ValueError("Cannot save image because frame is None.")
    
    path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv.imwrite(str(path), frame)

    if not success:
        raise ValueError(f"Failed to save image to: {output_path}")

    return success


def save_debug_frames(frames_with_timestamps: list[dict], output_dir: str) -> None:
    if not frames_with_timestamps:
        raise ValueError("frames_with_timestamps is empty.")

    total_frames = len(frames_with_timestamps)

    first_idx = 0
    middle_idx = total_frames // 2
    last_idx = total_frames - 1

    selected_frames = [
        ("first", frames_with_timestamps[first_idx]),
        ("middle", frames_with_timestamps[middle_idx]),
        ("last", frames_with_timestamps[last_idx]),
    ]

    for label, frame_record in selected_frames:
        frame_index = frame_record["frame_index"]
        timestamp = frame_record["timestamp"]
        frame = frame_record["frame"]

        filename = f"{label}_idx_{frame_index}_t_{timestamp:.3f}.jpg"
        output_path = Path(output_dir) / filename

        save_frame_image(frame, output_path)


