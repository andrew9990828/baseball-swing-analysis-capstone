"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: main.py
Description:
    Main entry point for the baseball swing analysis capstone project.
    This file is used to run selected project modules during development.

Last Updated: 5/18/26

Notes:
    Currently running Module 5 feature extraction using cleaned pose data
    from Module 3 and detected swing events from Module 4.
"""

from src.features.feature_pipeline import run_feature_pipeline


def main():
    cleaned_pose_path = "data/processed/pose/mike_trout_swing_01_pose_cleaned.npy"
    events_path = "data/processed/events/mike_trout_swing_01_events.json"
    output_path = "data/processed/features/mike_trout_swing_01_features.json"

    features = run_feature_pipeline(
        cleaned_pose_path=cleaned_pose_path,
        events_path=events_path,
        output_path=output_path,
    )

    print("\nModule 5 Feature Extraction Complete")
    print("-----------------------------------")

    for feature_name, feature_value in features.items():
        print(f"{feature_name}: {feature_value}")

    print(f"\nSaved features to: {output_path}")


if __name__ == "__main__":
    main()