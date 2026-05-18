"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: feature_pipeline.py
Description:
    Defines the feature extraction pipeline for Module 5.
    This module loads cleaned pose landmarks and detected swing events,
    runs the SwingFeatureExtractor, and saves the calculated swing features.

Last Updated: 5/18/26

Notes:
    This pipeline connects Module 3 cleaned landmark output with Module 4
    event detection output. The resulting features are saved as a JSON file
    so later feedback modules can interpret the swing metrics.
"""

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np

from src.features.feature_extractor import SwingFeatureExtractor


def load_cleaned_landmarks(cleaned_pose_path: str) -> np.ndarray:
    """
    Load cleaned landmark data from Module 3.

    Args:
        cleaned_pose_path:
            Path to the cleaned landmark .npy file.

    Returns:
        Cleaned landmark array with shape:
        (num_frames, 33, 4)
    """
    return np.load(cleaned_pose_path)


def load_events(events_path: str) -> Dict[str, int]:
    """
    Load detected swing events from Module 4.

    Args:
        events_path:
            Path to the events JSON file.

    Returns:
        Dictionary containing detected swing event frame indices.
    """
    with open(events_path, "r", encoding="utf-8") as file:
        return json.load(file)


def save_features(features: Dict[str, Any], output_path: str) -> None:
    """
    Save extracted swing features to a JSON file.

    Args:
        features:
            Feature dictionary to save.

        output_path:
            Destination JSON path.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as file:
        json.dump(features, file, indent=4)


def run_feature_pipeline(
    cleaned_pose_path: str,
    events_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Run the full Module 5 feature extraction pipeline.

    Args:
        cleaned_pose_path:
            Path to Module 3 cleaned landmark output.

        events_path:
            Path to Module 4 detected events JSON.

        output_path:
            Path where extracted features should be saved.

    Returns:
        Dictionary of extracted swing features.
    """
    landmarks = load_cleaned_landmarks(cleaned_pose_path)
    events = load_events(events_path)

    extractor = SwingFeatureExtractor(landmarks, events)
    features = extractor.extract_all_features().to_dict()

    save_features(features, output_path)

    return features