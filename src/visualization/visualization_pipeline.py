"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: visualization_pipeline.py
Description:
    Defines the visualization pipeline for Module 7.
    This module loads cleaned landmarks, detected events, and extracted
    features, then creates visual proof assets for the swing analysis system.

Last Updated: 5/25/26

Notes:
    This pipeline connects previous module outputs to simple visualization
    methods. The output images are meant to support debugging, review, and
    final report assets.
"""

import json
from typing import Dict, Any

import numpy as np

from src.visualization.visualizer import SwingVisualizer


def load_cleaned_landmarks(cleaned_pose_path: str) -> np.ndarray:
    """
    Load cleaned landmark data from Module 3.

    Args:
        cleaned_pose_path:
            Path to cleaned landmark .npy file.

    Returns:
        Cleaned landmark array with shape (num_frames, 33, 4).
    """
    return np.load(cleaned_pose_path)


def load_events(events_path: str) -> Dict[str, int]:
    """
    Load detected swing events from Module 4.

    Args:
        events_path:
            Path to events JSON file.

    Returns:
        Dictionary containing event frame indices.
    """
    with open(events_path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_features(features_path: str) -> Dict[str, Any]:
    """
    Load extracted swing features from Module 5.

    Args:
        features_path:
            Path to features JSON file.

    Returns:
        Dictionary containing extracted swing features.
    """
    with open(features_path, "r", encoding="utf-8") as file:
        return json.load(file)


def run_visualization_pipeline(
    video_path: str,
    cleaned_pose_path: str,
    events_path: str,
    features_path: str,
    output_dir: str,
) -> None:
    """
    Run the full Module 7 visualization pipeline.

    Args:
        video_path:
            Path to the original swing video.

        cleaned_pose_path:
            Path to Module 3 cleaned landmark output.

        events_path:
            Path to Module 4 event JSON output.

        features_path:
            Path to Module 5 feature JSON output.

        output_dir:
            Directory where visualization outputs should be saved.
    """
    landmarks = load_cleaned_landmarks(cleaned_pose_path)
    events = load_events(events_path)
    features = load_features(features_path)

    visualizer = SwingVisualizer(
        landmarks=landmarks,
        events=events,
        features=features,
        output_dir=output_dir,
    )

    visualizer.create_all_visualizations(video_path)