"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: event_detector.py
Description:
    Defines the swing event detection logic for Module 4.
    This module uses cleaned pose landmarks to estimate rough swing events
    such as movement start, peak hand speed, and contact proxy.

Last Updated: 5/7/26

Notes:
    This is a v1 event detector. Since the bat and ball are not tracked yet,
    contact is treated as a proxy based on body movement timing.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class SwingEvents:
    """
    Container for important detected swing event frames.
    """

    movement_start: int
    peak_hand_speed: int
    contact_proxy: int


class SwingEventDetector:
    """
    Detects rough swing events from cleaned landmark data.
    """

    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(self, movement_threshold_ratio: float = 0.25, contact_offset: int = 3):
        """
        Args:
            movement_threshold_ratio:
                Ratio of max hand speed used to decide when movement starts.

            contact_offset:
                Number of frames after peak hand speed used as a rough contact proxy.
        """
        self.movement_threshold_ratio = movement_threshold_ratio
        self.contact_offset = contact_offset

    def extract_landmark_xy(self, landmarks: np.ndarray, landmark_index: int) -> np.ndarray:
        """
        Extract x/y coordinates for one landmark across all frames.

        Args:
            landmarks:
                Shape: (num_frames, num_landmarks, 4)

            landmark_index:
                MediaPipe landmark index.

        Returns:
            Shape: (num_frames, 2)
        """
        return landmarks[:, landmark_index, 0:2]

    def calculate_frame_speeds(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Calculate frame-to-frame movement speed for one landmark trajectory.

        Speed here is a proxy based on distance moved between frames.
        """
        diffs = np.diff(trajectory, axis=0)
        speeds = np.linalg.norm(diffs, axis=1)

        # Add a zero at the start so speed array matches number of frames.
        speeds = np.insert(speeds, 0, 0.0)

        return speeds

    def calculate_hand_speed_proxy(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate a rough hand speed proxy using left and right wrist movement.

        Since handedness is not handled yet, this uses the max speed between wrists.
        """
        left_wrist_xy = self.extract_landmark_xy(landmarks, self.LEFT_WRIST)
        right_wrist_xy = self.extract_landmark_xy(landmarks, self.RIGHT_WRIST)

        left_speed = self.calculate_frame_speeds(left_wrist_xy)
        right_speed = self.calculate_frame_speeds(right_wrist_xy)

        hand_speed_proxy = np.maximum(left_speed, right_speed)

        return hand_speed_proxy

    def detect_movement_start(self, hand_speed: np.ndarray) -> int:
        """
        Detect the first frame where hand speed rises above a threshold.
        """
        max_speed = np.max(hand_speed)

        if max_speed == 0:
            return 0

        threshold = max_speed * self.movement_threshold_ratio

        for frame_index, speed in enumerate(hand_speed):
            if speed >= threshold:
                return frame_index

        return 0

    def detect_peak_hand_speed(self, hand_speed: np.ndarray) -> int:
        """
        Detect the frame where hand speed is highest.
        """
        return int(np.argmax(hand_speed))

    def detect_contact_proxy(self, peak_hand_speed_frame: int, num_frames: int) -> int:
        """
        Estimate contact as a few frames after peak hand speed.

        This is only a v1 proxy because bat/ball contact is not directly tracked.
        """
        contact_frame = peak_hand_speed_frame + self.contact_offset

        if contact_frame >= num_frames:
            contact_frame = num_frames - 1

        return contact_frame

    def detect_events(self, landmarks: np.ndarray) -> SwingEvents:
        """
        Detect all v1 swing events.

        Args:
            landmarks:
                Cleaned landmark array with shape:
                (num_frames, 33, 4)

        Returns:
            SwingEvents dataclass.
        """
        hand_speed = self.calculate_hand_speed_proxy(landmarks)

        movement_start = self.detect_movement_start(hand_speed)
        peak_hand_speed = self.detect_peak_hand_speed(hand_speed)
        contact_proxy = self.detect_contact_proxy(
            peak_hand_speed_frame=peak_hand_speed,
            num_frames=landmarks.shape[0]
        )

        return SwingEvents(
            movement_start=movement_start,
            peak_hand_speed=peak_hand_speed,
            contact_proxy=contact_proxy
        )