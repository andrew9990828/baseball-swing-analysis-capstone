"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: main.py
Description:
    Batch entry point for the baseball swing analysis capstone project.

    This script runs the full v1 pipeline over every video in a raw video
    folder, then aggregates extracted features into a professional baseline.

Pipeline:
    raw video
    -> pose extraction
    -> landmark processing
    -> event detection
    -> feature extraction
    -> feedback generation
    -> visualization output
    -> evaluation report
    -> aggregate feature table
    -> pro baseline JSON
    -> aggregate metric plots
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from src.pose.pose_pipeline import extract_pose_from_video
from src.processing.landmark_processor import process_landmarks
from src.events.event_pipeline import run_event_detection_pipeline
from src.features.feature_pipeline import run_feature_pipeline
from src.feedback.feedback_engine import FeedbackEngine
from src.visualization.visualization_pipeline import run_visualization_pipeline
from src.evaluation.evaluator import SwingEvaluator


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "pros"

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_POSE_DIR = PROCESSED_DIR / "pose"
PROCESSED_EVENTS_DIR = PROCESSED_DIR / "events"
PROCESSED_FEATURES_DIR = PROCESSED_DIR / "features"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
DEBUG_POSE_DIR = OUTPUTS_DIR / "debug_pose"
VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
REPORT_DIR = OUTPUTS_DIR / "report"
AGGREGATE_DIR = OUTPUTS_DIR / "aggregate"

BASELINE_FEATURES = [
    "head_movement_start_to_contact",
    "hand_path_start_to_contact",
    "shoulder_angle_change_start_to_contact",
    "frames_start_to_contact",
    "frames_start_to_peak_hand_speed",
    "frames_peak_hand_speed_to_contact",
]

MOVEMENT_METRICS = [
    "head_movement_start_to_contact",
    "hand_path_start_to_contact",
]

ROTATION_METRICS = [
    "shoulder_angle_change_start_to_contact",
]

TIMING_METRICS = [
    "frames_start_to_contact",
    "frames_start_to_peak_hand_speed",
    "frames_peak_hand_speed_to_contact",
]


def get_video_paths(raw_dir: Path) -> list[Path]:
    """
    Return all .mp4 videos in the selected raw video folder.
    """
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw video directory not found: {raw_dir}")

    video_paths = sorted(raw_dir.glob("*.mp4"))

    if not video_paths:
        raise FileNotFoundError(f"No .mp4 videos found in: {raw_dir}")

    return video_paths


def build_paths(video_path: Path) -> dict[str, Path]:
    """
    Build every input/output path for a single video based on its filename stem.

    Example:
        data/raw/pros/shohei_ohtani_01.mp4

    video_id:
        shohei_ohtani_01
    """
    video_id = video_path.stem

    return {
        "video_id": Path(video_id),
        "video_path": video_path,
        "raw_pose_path": PROCESSED_POSE_DIR / f"{video_id}_pose_raw.npz",
        "cleaned_pose_path": PROCESSED_POSE_DIR / f"{video_id}_pose_cleaned.npy",
        "events_path": PROCESSED_EVENTS_DIR / f"{video_id}_events.json",
        "features_path": PROCESSED_FEATURES_DIR / f"{video_id}_features.json",
        "debug_pose_dir": DEBUG_POSE_DIR / video_id,
        "visualization_dir": VISUALIZATIONS_DIR / video_id,
        "evaluation_report_path": REPORT_DIR / f"{video_id}_evaluation.md",
    }


def load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file.
    """
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_json(data: dict[str, Any], path: Path) -> None:
    """
    Save a dictionary to JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def run_single_video(
    video_path: Path,
    group_label: str,
    skip_existing: bool = False,
    max_frames: int | None = None,
    debug_every_n_frames: int = 15,
    smoothing_window: int = 3,
) -> dict[str, Any]:
    """
    Run the full v1 pipeline for one video.
    """
    paths = build_paths(video_path)
    video_id = paths["video_id"].name

    print("\n" + "=" * 80)
    print(f"Processing video: {video_id}")
    print("=" * 80)

    raw_pose_path = paths["raw_pose_path"]
    cleaned_pose_path = paths["cleaned_pose_path"]
    events_path = paths["events_path"]
    features_path = paths["features_path"]
    debug_pose_dir = paths["debug_pose_dir"]
    visualization_dir = paths["visualization_dir"]
    evaluation_report_path = paths["evaluation_report_path"]

    try:
        if skip_existing and raw_pose_path.exists():
            print(f"Skipping pose extraction. Found: {raw_pose_path}")
        else:
            extract_pose_from_video(
                video_path=str(video_path),
                output_path=str(raw_pose_path),
                debug_dir=str(debug_pose_dir),
                max_frames=max_frames,
                debug_every_n_frames=debug_every_n_frames,
            )

        if skip_existing and cleaned_pose_path.exists():
            print(f"Skipping landmark processing. Found: {cleaned_pose_path}")
        else:
            process_landmarks(
                input_path=raw_pose_path,
                output_path=cleaned_pose_path,
                smoothing_window=smoothing_window,
            )

        if skip_existing and events_path.exists():
            print(f"Skipping event detection. Found: {events_path}")
        else:
            run_event_detection_pipeline(
                input_path=cleaned_pose_path,
                output_path=events_path,
            )

        if skip_existing and features_path.exists():
            print(f"Skipping feature extraction. Found: {features_path}")
            features = load_json(features_path)
        else:
            features = run_feature_pipeline(
                cleaned_pose_path=str(cleaned_pose_path),
                events_path=str(events_path),
                output_path=str(features_path),
            )

        feedback_engine = FeedbackEngine(features)
        feedback_output = feedback_engine.generate_feedback()

        run_visualization_pipeline(
            video_path=str(video_path),
            cleaned_pose_path=str(cleaned_pose_path),
            events_path=str(events_path),
            features_path=str(features_path),
            output_dir=str(visualization_dir),
        )

        events = load_json(events_path)

        evaluator = SwingEvaluator(
            sample_name=video_id,
            events=events,
            features=features,
            feedback_output=feedback_output,
            visualization_dir=str(visualization_dir),
            output_report_path=str(evaluation_report_path),
        )

        evaluator.run()

        print(f"\nFinished: {video_id}")
        print(f"Features: {features_path}")
        print(f"Visualizations: {visualization_dir}")
        print(f"Evaluation report: {evaluation_report_path}")

        return {
            "video_id": video_id,
            "group": group_label,
            "video_path": str(video_path),
            "status": "success",
            "features_path": str(features_path),
            "events_path": str(events_path),
            "evaluation_report_path": str(evaluation_report_path),
            **features,
        }

    except Exception as error:
        print(f"\nFAILED: {video_id}")
        print(f"Error: {error}")

        return {
            "video_id": video_id,
            "group": group_label,
            "video_path": str(video_path),
            "status": "failed",
            "error": str(error),
        }


def is_number(value: Any) -> bool:
    """
    Return True only if a value can safely be converted to a finite float.

    This rejects:
        - None
        - strings that cannot become numbers
        - NaN
        - infinity
    """
    try:
        number = float(value)
        return np.isfinite(number)
    except (TypeError, ValueError):
        return False


def write_feature_table(results: list[dict[str, Any]], output_path: Path) -> None:
    """
    Write all per-video results into one CSV feature table.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_keys: set[str] = set()

    for result in results:
        all_keys.update(result.keys())

    preferred_first = [
        "video_id",
        "group",
        "status",
        "video_path",
        "features_path",
        "events_path",
        "evaluation_report_path",
        "error",
    ]

    remaining_keys = sorted(key for key in all_keys if key not in preferred_first)
    fieldnames = [key for key in preferred_first if key in all_keys] + remaining_keys

    with open(output_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved aggregate feature table to: {output_path}")


def compute_baseline(
    results: list[dict[str, Any]],
    group_label: str,
    feature_names: list[str],
) -> dict[str, Any]:
    """
    Compute median/IQR baseline stats for successful videos in the target group.
    """
    successful_group_results = [
        result
        for result in results
        if result.get("status") == "success" and result.get("group") == group_label
    ]

    baseline: dict[str, Any] = {
        "group": group_label,
        "sample_count": len(successful_group_results),
        "features": {},
        "notes": [
            "Baseline uses median and interquartile range because the sample size is small.",
            "Values are pose-derived metrics, not ground-truth biomechanics measurements.",
            "This should be described as a professional sample baseline, not an ideal swing model.",
            "Hip drift is intentionally excluded because current cleaned landmarks are normalized around the hip center.",
        ],
    }

    for feature_name in feature_names:
        values = [
            float(result[feature_name])
            for result in successful_group_results
            if feature_name in result and is_number(result[feature_name])
        ]

        if not values:
            continue

        values_array = np.array(values, dtype=float)

        baseline["features"][feature_name] = {
            "count": int(len(values_array)),
            "median": float(np.median(values_array)),
            "q1": float(np.percentile(values_array, 25)),
            "q3": float(np.percentile(values_array, 75)),
            "iqr": float(np.percentile(values_array, 75) - np.percentile(values_array, 25)),
            "min": float(np.min(values_array)),
            "max": float(np.max(values_array)),
        }

    return baseline


def plot_metric_group(
    results: list[dict[str, Any]],
    metrics: list[str],
    title: str,
    output_path: Path,
) -> None:
    """
    Plot selected metrics across all successful videos.

    This keeps movement, rotation, and timing metrics separate so we do not mix
    unrelated units on one chart.
    """
    successful_results = [
        result for result in results if result.get("status") == "success"
    ]

    if not successful_results:
        print(f"No successful results available for plot: {title}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    video_ids = [result["video_id"] for result in successful_results]

    plt.figure(figsize=(12, 6))

    plotted_any_metric = False

    for metric in metrics:
        values = []

        for result in successful_results:
            if metric in result and is_number(result[metric]):
                values.append(float(result[metric]))
            else:
                values.append(np.nan)

        if all(np.isnan(value) for value in values):
            continue

        plotted_any_metric = True
        plt.plot(video_ids, values, marker="o", label=metric)

        clean_values = [value for value in values if not np.isnan(value)]

        if clean_values:
            median_value = float(np.median(clean_values))
            plt.axhline(
                median_value,
                linestyle="--",
                linewidth=1,
                label=f"{metric} median",
            )

    if not plotted_any_metric:
        plt.close()
        print(f"No plottable metrics found for: {title}")
        return

    plt.title(title)
    plt.xlabel("Video")
    plt.ylabel("Metric value")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved plot to: {output_path}")


def write_batch_summary(
    results: list[dict[str, Any]],
    baseline: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save a JSON summary of the batch run.
    """
    successful = [result for result in results if result.get("status") == "success"]
    failed = [result for result in results if result.get("status") == "failed"]

    summary = {
        "total_videos": len(results),
        "successful_videos": len(successful),
        "failed_videos": len(failed),
        "failed_video_ids": [result["video_id"] for result in failed],
        "baseline": baseline,
    }

    save_json(summary, output_path)
    print(f"Saved batch summary to: {output_path}")


def run_batch(
    raw_dir: Path,
    group_label: str,
    skip_existing: bool,
    max_frames: int | None,
    debug_every_n_frames: int,
    smoothing_window: int,
) -> None:
    """
    Run the full batch pipeline and aggregate outputs.
    """
    video_paths = get_video_paths(raw_dir)

    print("\nFound videos:")
    for video_path in video_paths:
        print(f"- {video_path.name}")

    results = []

    for video_path in video_paths:
        result = run_single_video(
            video_path=video_path,
            group_label=group_label,
            skip_existing=skip_existing,
            max_frames=max_frames,
            debug_every_n_frames=debug_every_n_frames,
            smoothing_window=smoothing_window,
        )
        results.append(result)

    feature_table_path = AGGREGATE_DIR / f"{group_label}_feature_table.csv"
    baseline_path = AGGREGATE_DIR / f"{group_label}_baseline.json"
    batch_summary_path = AGGREGATE_DIR / f"{group_label}_batch_summary.json"

    write_feature_table(results, feature_table_path)

    baseline = compute_baseline(
        results=results,
        group_label=group_label,
        feature_names=BASELINE_FEATURES,
    )

    save_json(baseline, baseline_path)
    print(f"Saved baseline to: {baseline_path}")

    plot_metric_group(
        results=results,
        metrics=MOVEMENT_METRICS,
        title="Professional Baseline - Movement Metrics",
        output_path=AGGREGATE_DIR / f"{group_label}_movement_metrics.png",
    )

    plot_metric_group(
        results=results,
        metrics=ROTATION_METRICS,
        title="Professional Baseline - Rotation Metrics",
        output_path=AGGREGATE_DIR / f"{group_label}_rotation_metrics.png",
    )

    plot_metric_group(
        results=results,
        metrics=TIMING_METRICS,
        title="Professional Baseline - Timing Metrics",
        output_path=AGGREGATE_DIR / f"{group_label}_timing_metrics.png",
    )

    write_batch_summary(
        results=results,
        baseline=baseline,
        output_path=batch_summary_path,
    )

    print("\n" + "=" * 80)
    print("Batch pipeline complete")
    print("=" * 80)
    print(f"Videos processed: {len(results)}")
    print(f"Feature table: {feature_table_path}")
    print(f"Baseline JSON: {baseline_path}")
    print(f"Aggregate outputs: {AGGREGATE_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseball swing analysis pipeline over a folder of videos."
    )

    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help="Folder containing raw .mp4 swing videos.",
    )

    parser.add_argument(
        "--group-label",
        type=str,
        default="pro",
        help="Group label to attach to every processed video.",
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing intermediate files when possible.",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process per video.",
    )

    parser.add_argument(
        "--debug-every-n-frames",
        type=int,
        default=15,
        help="Save one pose debug frame every N frames.",
    )

    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=3,
        help="Moving-average smoothing window for landmark processing.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_batch(
        raw_dir=args.raw_dir,
        group_label=args.group_label,
        skip_existing=args.skip_existing,
        max_frames=args.max_frames,
        debug_every_n_frames=args.debug_every_n_frames,
        smoothing_window=args.smoothing_window,
    )


if __name__ == "__main__":
    main()