"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: event_pipeline.py
Description:
    Pipeline for running Module 4 event detection on cleaned landmark data.
    This module loads cleaned landmarks, runs the swing event detector,
    and saves detected event frame indices to a JSON file.

Last Updated: 5/7/26

Notes:
    This pipeline connects Module 3 cleaned landmark output to Module 4
    swing event detection.
"""

from pathlib import Path
import json
import numpy as np

from src.events.event_detector import SwingEventDetector, SwingEvents


def load_cleaned_landmarks(input_path: str | Path) -> np.ndarray:
    """
    Load cleaned landmark data from Module 3.
    """
    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Cleaned landmark file not found: {input_path}")

    landmarks = np.load(input_path)

    return landmarks


def swing_events_to_dict(events: SwingEvents) -> dict:
    """
    Convert SwingEvents dataclass into a normal dictionary for JSON saving.
    """
    return {
        "movement_start": events.movement_start,
        "peak_hand_speed": events.peak_hand_speed,
        "contact_proxy": events.contact_proxy
    }


def save_events(events: SwingEvents, output_path: str | Path) -> None:
    """
    Save detected swing events to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    events_dict = swing_events_to_dict(events)

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(events_dict, file, indent=4)

    print(f"Saved swing events to: {output_path}")


def run_event_detection_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    movement_threshold_ratio: float = 0.25,
    contact_offset: int = 3
) -> SwingEvents:
    """
    Full Module 4 event detection pipeline.

    Steps:
        1. Load cleaned landmarks from Module 3
        2. Create event detector
        3. Detect rough swing events
        4. Save event results
    """
    landmarks = load_cleaned_landmarks(input_path)

    print("Loaded cleaned landmarks:")
    print("Shape:", landmarks.shape)

    detector = SwingEventDetector(
        movement_threshold_ratio=movement_threshold_ratio,
        contact_offset=contact_offset
    )

    events = detector.detect_events(landmarks)

    print("Detected swing events:")
    print(events)

    save_events(events, output_path)

    return events