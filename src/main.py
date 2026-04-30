from pathlib import Path

from src.processing.landmark_processor import process_landmarks


def main():
    input_path = Path("data/processed/pose/mike_trout_swing_01_pose_raw.npz")
    output_path = Path("data/processed/pose/mike_trout_swing_01_pose_cleaned.npy")

    process_landmarks(
        input_path=input_path,
        output_path=output_path,
        smoothing_window=3
    )


if __name__ == "__main__":
    main()