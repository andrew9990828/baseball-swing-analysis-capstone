"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: main.py
Description:
    Main entry point for the baseball swing analysis capstone project.

Last Updated: 4/27/26
"""

from src.processing.pose_pipeline import extract_pose_from_video


def main() -> None:
    """
    Run the current Module 2 pose extraction pipeline.
    """

    video_path = "data/raw/mike_trout_swing_01.mp4"
    output_path = "data/processed/pose/mike_trout_swing_01_pose_raw.npz"
    debug_dir = "outputs/debug_pose"

    extract_pose_from_video(
        video_path=video_path,
        output_path=output_path,
        debug_dir=debug_dir,
        max_frames=None,
        debug_every_n_frames=10,
    )


if __name__ == "__main__":
    main()