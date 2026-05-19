"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: main.py
Description:
    Main entry point for the baseball swing analysis capstone project.
    This file is used to run selected project modules during development.

Last Updated: 5/18/26

Notes:
    Currently running Module 5 feature extraction and Module 6 feedback
    generation using the Mike Trout sample swing.
"""

from src.features.feature_pipeline import run_feature_pipeline
from src.feedback.feedback_engine import FeedbackEngine


def main():
    cleaned_pose_path = "data/processed/pose/mike_trout_swing_01_pose_cleaned.npy"
    events_path = "data/processed/events/mike_trout_swing_01_events.json"
    features_output_path = "data/processed/features/mike_trout_swing_01_features.json"

    features = run_feature_pipeline(
        cleaned_pose_path=cleaned_pose_path,
        events_path=events_path,
        output_path=features_output_path,
    )

    print("\nModule 5 Feature Extraction Complete")
    print("-----------------------------------")

    for feature_name, feature_value in features.items():
        print(f"{feature_name}: {feature_value}")

    print(f"\nSaved features to: {features_output_path}")

    feedback_engine = FeedbackEngine(features)
    feedback_output = feedback_engine.generate_feedback()

    print("\nModule 6 Feedback Engine Complete")
    print("--------------------------------")

    print("\nSummary:")
    print(feedback_output["summary"])

    print("\nFeedback:")
    for item in feedback_output["feedback"]:
        print(f"\nMetric: {item['metric']}")
        print(f"Status: {item['status']}")
        print(f"Statement: {item['statement']}")
        print(f"Evidence: {item['evidence']}")
        print(f"Confidence: {item['confidence']}")

    print("\nWarnings:")
    for warning in feedback_output["warnings"]:
        print(f"\nMetric: {warning['metric']}")
        print(f"Warning: {warning['warning']}")


if __name__ == "__main__":
    main()