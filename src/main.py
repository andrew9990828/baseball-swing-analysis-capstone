"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: main.py
Description:
    Main entry point for the baseball swing analysis capstone project.
    This file is used to run selected project modules during development.

Last Updated: 5/26/26

Notes:
    Currently running Module 5 feature extraction, Module 6 feedback
    generation, Module 7 visualization, and Module 8 evaluation using the
    Mike Trout sample swing.
"""

import json

from src.features.feature_pipeline import run_feature_pipeline
from src.feedback.feedback_engine import FeedbackEngine
from src.visualization.visualization_pipeline import run_visualization_pipeline
from src.evaluation.evaluator import SwingEvaluator


def main():
    sample_name = "mike_trout_swing_01"

    video_path = "data/raw/mike_trout_swing_01.mp4"
    cleaned_pose_path = "data/processed/pose/mike_trout_swing_01_pose_cleaned.npy"
    events_path = "data/processed/events/mike_trout_swing_01_events.json"
    features_output_path = "data/processed/features/mike_trout_swing_01_features.json"

    visualization_output_dir = "outputs/visualizations/mike_trout_swing_01"
    evaluation_report_path = "outputs/report/mike_trout_swing_01_evaluation.md"

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

    run_visualization_pipeline(
        video_path=video_path,
        cleaned_pose_path=cleaned_pose_path,
        events_path=events_path,
        features_path=features_output_path,
        output_dir=visualization_output_dir,
    )

    print("\nModule 7 Visualization Complete")
    print("-------------------------------")
    print(f"Saved visualizations to: {visualization_output_dir}")

    with open(events_path, "r", encoding="utf-8") as file:
        events = json.load(file)

    evaluator = SwingEvaluator(
        sample_name=sample_name,
        events=events,
        features=features,
        feedback_output=feedback_output,
        visualization_dir=visualization_output_dir,
        output_report_path=evaluation_report_path,
    )

    saved_report_path = evaluator.run()

    print("\nModule 8 Evaluation Complete")
    print("----------------------------")
    print(f"Saved evaluation report to: {saved_report_path}")


if __name__ == "__main__":
    main()