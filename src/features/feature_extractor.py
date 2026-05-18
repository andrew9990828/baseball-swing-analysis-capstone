"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: feature_extractor.py
Description:
    Defines the swing feature extraction logic for Module 5.
    This module uses cleaned pose landmarks and previously detected swing
    events to calculate interpretable mechanics metrics such as head movement,
    hand path distance, hip drift, shoulder rotation proxy, and timing features.

Last Updated: 5/18/26

Notes:
    This is a v1 feature extractor. Since the bat and ball are not tracked yet,
    all metrics are based on body landmark movement and event proxies detected
    in earlier modules.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any

import numpy as np


@dataclass
class SwingFeatures:
    """
    Container for the main extracted swing features.

    All distance-based values are currently in normalized landmark units.
    Angle-based values are in degrees.
    Frame-based values are in frame counts.
    """

    head_movement_start_to_contact: float
    hand_path_start_to_contact: float
    hip_drift_start_to_contact: float
    shoulder_angle_change_start_to_contact: float
    frames_start_to_contact: int
    frames_start_to_peak_hand_speed: int
    frames_peak_hand_speed_to_contact: int

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the feature container into a dictionary for saving as JSON.
        """
        return asdict(self)


class SwingFeatureExtractor:
    """
    Calculates interpretable swing mechanics features from cleaned landmarks
    and detected swing events.
    """

    # MediaPipe landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self, landmarks: np.ndarray, events: Dict[str, int]):
        """
        Initialize the feature extractor.

        Args:
            landmarks:
                Cleaned landmark array with shape:
                (num_frames, 33, 4)

                Expected values per landmark:
                x, y, z, visibility

            events:
                Dictionary of detected swing events from Module 4.
                Expected keys:
                - movement_start
                - peak_hand_speed
                - contact_proxy
        """
        self.landmarks = landmarks
        self.events = events

        self.movement_start = int(events["movement_start"])
        self.peak_hand_speed = int(events["peak_hand_speed"])
        self.contact_proxy = int(events["contact_proxy"])

    def _get_xy(self, frame_index: int, landmark_index: int) -> np.ndarray:
        """
        Get the x, y position for a landmark at a specific frame.

        Args:
            frame_index:
                Frame number to read from.

            landmark_index:
                MediaPipe landmark index.

        Returns:
            NumPy array containing [x, y].
        """
        return self.landmarks[frame_index, landmark_index, :2]

    def _calculate_distance(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Calculate Euclidean distance between two 2D points.

        Args:
            point_a:
                First point as [x, y].

            point_b:
                Second point as [x, y].

        Returns:
            Distance between the two points.
        """
        return float(np.linalg.norm(point_b - point_a))

    def _calculate_path_distance(
        self,
        landmark_index: int,
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Calculate total movement path distance for one landmark over time.

        Args:
            landmark_index:
                MediaPipe landmark index to track.

            start_frame:
                Starting frame index.

            end_frame:
                Ending frame index.

        Returns:
            Total path distance across consecutive frames.
        """
        total_distance = 0.0

        for frame_index in range(start_frame, end_frame):
            current_point = self._get_xy(frame_index, landmark_index)
            next_point = self._get_xy(frame_index + 1, landmark_index)

            total_distance += self._calculate_distance(current_point, next_point)

        return float(total_distance)

    def _calculate_midpoint(
        self,
        frame_index: int,
        landmark_a: int,
        landmark_b: int,
    ) -> np.ndarray:
        """
        Calculate midpoint between two landmarks at a specific frame.

        Args:
            frame_index:
                Frame number to read from.

            landmark_a:
                First MediaPipe landmark index.

            landmark_b:
                Second MediaPipe landmark index.

        Returns:
            Midpoint as [x, y].
        """
        point_a = self._get_xy(frame_index, landmark_a)
        point_b = self._get_xy(frame_index, landmark_b)

        return (point_a + point_b) / 2.0

    def _calculate_angle_degrees(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Calculate the angle of the line from point_a to point_b.

        Args:
            point_a:
                First point as [x, y].

            point_b:
                Second point as [x, y].

        Returns:
            Angle in degrees.
        """
        delta = point_b - point_a
        angle_radians = np.arctan2(delta[1], delta[0])

        return float(np.degrees(angle_radians))

    def calculate_head_movement(self) -> float:
        """
        Calculate head movement from movement start to contact proxy.

        Uses the nose landmark as a simple v1 head movement proxy.

        Returns:
            Distance from nose position at movement start to nose position
            at contact proxy.
        """
        start_head = self._get_xy(self.movement_start, self.NOSE)
        contact_head = self._get_xy(self.contact_proxy, self.NOSE)

        return self._calculate_distance(start_head, contact_head)

    def calculate_hand_path_distance(self) -> float:
        """
        Calculate hand path distance from movement start to contact proxy.

        Uses the average path distance of the left and right wrists.

        Returns:
            Average wrist path distance.
        """
        left_wrist_path = self._calculate_path_distance(
            self.LEFT_WRIST,
            self.movement_start,
            self.contact_proxy,
        )

        right_wrist_path = self._calculate_path_distance(
            self.RIGHT_WRIST,
            self.movement_start,
            self.contact_proxy,
        )

        return float((left_wrist_path + right_wrist_path) / 2.0)

    def calculate_hip_drift(self) -> float:
        """
        Calculate hip center drift from movement start to contact proxy.

        Uses midpoint between left hip and right hip.

        Returns:
            Hip center distance from movement start to contact proxy.
        """
        start_hip_center = self._calculate_midpoint(
            self.movement_start,
            self.LEFT_HIP,
            self.RIGHT_HIP,
        )

        contact_hip_center = self._calculate_midpoint(
            self.contact_proxy,
            self.LEFT_HIP,
            self.RIGHT_HIP,
        )

        return self._calculate_distance(start_hip_center, contact_hip_center)

    def calculate_shoulder_angle_change(self) -> float:
        """
        Calculate shoulder line angle change from movement start to contact proxy.

        Uses the line from left shoulder to right shoulder as a v1 shoulder
        rotation proxy.

        Returns:
            Absolute shoulder angle change in degrees.
        """
        start_left_shoulder = self._get_xy(self.movement_start, self.LEFT_SHOULDER)
        start_right_shoulder = self._get_xy(self.movement_start, self.RIGHT_SHOULDER)

        contact_left_shoulder = self._get_xy(self.contact_proxy, self.LEFT_SHOULDER)
        contact_right_shoulder = self._get_xy(self.contact_proxy, self.RIGHT_SHOULDER)

        start_angle = self._calculate_angle_degrees(
            start_left_shoulder,
            start_right_shoulder,
        )

        contact_angle = self._calculate_angle_degrees(
            contact_left_shoulder,
            contact_right_shoulder,
        )

        return float(abs(contact_angle - start_angle))

    def calculate_timing_features(self) -> Dict[str, int]:
        """
        Calculate frame-based timing relationships between detected events.

        Returns:
            Dictionary containing timing differences in frames.
        """
        return {
            "frames_start_to_contact": self.contact_proxy - self.movement_start,
            "frames_start_to_peak_hand_speed": self.peak_hand_speed - self.movement_start,
            "frames_peak_hand_speed_to_contact": self.contact_proxy - self.peak_hand_speed,
        }

    def extract_all_features(self) -> SwingFeatures:
        """
        Run all feature calculations.

        Returns:
            SwingFeatures dataclass containing all v1 metrics.
        """
        timing_features = self.calculate_timing_features()

        return SwingFeatures(
            head_movement_start_to_contact=self.calculate_head_movement(),
            hand_path_start_to_contact=self.calculate_hand_path_distance(),
            hip_drift_start_to_contact=self.calculate_hip_drift(),
            shoulder_angle_change_start_to_contact=self.calculate_shoulder_angle_change(),
            frames_start_to_contact=timing_features["frames_start_to_contact"],
            frames_start_to_peak_hand_speed=timing_features[
                "frames_start_to_peak_hand_speed"
            ],
            frames_peak_hand_speed_to_contact=timing_features[
                "frames_peak_hand_speed_to_contact"
            ],
        )