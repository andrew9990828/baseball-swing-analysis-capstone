"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: visualizer.py
Description:
    Defines visualization methods for Module 7.
    This module creates plots and visual proof assets using cleaned pose
    landmarks, detected swing events, and extracted swing features.

Last Updated: 5/25/26

Notes:
    This is a v1 visualizer. The goal is not polished reporting yet.
    The goal is to create simple visual evidence that supports the
    extracted metrics and feedback.
"""

from pathlib import Path
from typing import Dict, Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


class SwingVisualizer:
    """
    Creates visual proof assets for the swing analysis pipeline.

    This class does not calculate new swing features.
    It visualizes existing data from earlier modules.
    """

    # MediaPipe landmark indices
    NOSE = 0
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    def __init__(
        self,
        landmarks: np.ndarray,
        events: Dict[str, int],
        features: Dict[str, Any],
        output_dir: str,
    ):
        """
        Initialize the visualizer.

        Args:
            landmarks:
                Cleaned landmark array with shape (num_frames, 33, 4).

            events:
                Dictionary of detected swing events.

            features:
                Dictionary of extracted swing features.

            output_dir:
                Directory where visualization outputs should be saved.
        """
        self.landmarks = landmarks
        self.events = events
        self.features = features
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_xy_series(self, landmark_index: int) -> np.ndarray:
        """
        Get x, y positions for one landmark across all frames.

        Args:
            landmark_index:
                MediaPipe landmark index.

        Returns:
            Array with shape (num_frames, 2).
        """
        return self.landmarks[:, landmark_index, :2]

    def _save_current_plot(self, filename: str) -> None:
        """
        Save the active matplotlib figure.

        Args:
            filename:
                Name of the output image file.
        """
        output_path = self.output_dir / filename
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    def plot_hand_path(self) -> None:
        """
        Create a 2D trajectory plot of left and right wrist movement.

        Output:
            hand_path_plot.png
        """
        left_wrist = self._get_xy_series(self.LEFT_WRIST)
        right_wrist = self._get_xy_series(self.RIGHT_WRIST)

        movement_start = self.events["movement_start"]
        contact_proxy = self.events["contact_proxy"]

        plt.figure(figsize=(8, 6))

        plt.plot(left_wrist[:, 0], left_wrist[:, 1], label="Left Wrist Path")
        plt.plot(right_wrist[:, 0], right_wrist[:, 1], label="Right Wrist Path")

        plt.scatter(
            left_wrist[movement_start, 0],
            left_wrist[movement_start, 1],
            label="Left Wrist Start",
            marker="o",
        )
        plt.scatter(
            left_wrist[contact_proxy, 0],
            left_wrist[contact_proxy, 1],
            label="Left Wrist Contact Proxy",
            marker="x",
        )

        plt.scatter(
            right_wrist[movement_start, 0],
            right_wrist[movement_start, 1],
            label="Right Wrist Start",
            marker="o",
        )
        plt.scatter(
            right_wrist[contact_proxy, 0],
            right_wrist[contact_proxy, 1],
            label="Right Wrist Contact Proxy",
            marker="x",
        )

        plt.title("Hand Path Trajectory")
        plt.xlabel("Normalized X Position")
        plt.ylabel("Normalized Y Position")
        plt.legend()
        plt.grid(True)

        self._save_current_plot("hand_path_plot.png")

    def plot_head_path(self) -> None:
        """
        Create a 2D trajectory plot of nose/head movement.

        Output:
            head_path_plot.png
        """
        nose = self._get_xy_series(self.NOSE)

        movement_start = self.events["movement_start"]
        contact_proxy = self.events["contact_proxy"]

        plt.figure(figsize=(8, 6))

        plt.plot(nose[:, 0], nose[:, 1], label="Nose / Head Path")

        plt.scatter(
            nose[movement_start, 0],
            nose[movement_start, 1],
            label="Movement Start",
            marker="o",
        )
        plt.scatter(
            nose[contact_proxy, 0],
            nose[contact_proxy, 1],
            label="Contact Proxy",
            marker="x",
        )

        plt.title("Head Movement Trajectory")
        plt.xlabel("Normalized X Position")
        plt.ylabel("Normalized Y Position")
        plt.legend()
        plt.grid(True)

        self._save_current_plot("head_path_plot.png")

    def plot_feature_summary(self) -> None:
        """
        Create a simple bar chart of selected Module 5 features.

        Output:
            feature_summary.png
        """
        selected_features = {
            "Head Movement": self.features["head_movement_start_to_contact"],
            "Hand Path": self.features["hand_path_start_to_contact"],
            "Shoulder Angle": self.features[
                "shoulder_angle_change_start_to_contact"
            ],
            "Start to Contact": self.features["frames_start_to_contact"],
        }

        names = list(selected_features.keys())
        values = list(selected_features.values())

        plt.figure(figsize=(10, 6))
        plt.bar(names, values)

        plt.title("Swing Feature Summary")
        plt.ylabel("Value")
        plt.xticks(rotation=20)
        plt.grid(axis="y")

        self._save_current_plot("feature_summary.png")

    def plot_timing_events(self) -> None:
        """
        Create a simple horizontal timeline of detected swing events.

        Output:
            timing_events.png
        """
        event_names = list(self.events.keys())
        event_frames = list(self.events.values())

        plt.figure(figsize=(10, 3))

        plt.hlines(y=1, xmin=0, xmax=max(event_frames), linewidth=2)

        for event_name, frame_index in self.events.items():
            plt.scatter(frame_index, 1)
            plt.text(
                frame_index,
                1.03,
                f"{event_name}\nFrame {frame_index}",
                ha="center",
                va="bottom",
            )

        plt.title("Detected Swing Event Timeline")
        plt.xlabel("Frame Index")
        plt.yticks([])
        plt.grid(axis="x")

        self._save_current_plot("timing_events.png")

    def save_key_event_frames(self, video_path: str) -> None:
        """
        Save raw video frames for the detected key swing events.

        Args:
            video_path:
                Path to the original swing video.

        Outputs:
            key_frame_movement_start.jpg
            key_frame_peak_hand_speed.jpg
            key_frame_contact_proxy.jpg
        """
        capture = cv2.VideoCapture(video_path)

        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        for event_name, frame_index in self.events.items():
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = capture.read()

            if not success:
                print(f"Warning: Could not read frame {frame_index} for {event_name}")
                continue

            label = f"{event_name} | frame {frame_index}"

            cv2.putText(
                frame,
                label,
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (255, 255, 255),
                3,
            )

            output_path = self.output_dir / f"key_frame_{event_name}.jpg"
            cv2.imwrite(str(output_path), frame)

        capture.release()

    def create_all_visualizations(self, video_path: str) -> None:
        """
        Create all v1 visualization outputs.

        Args:
            video_path:
                Path to the original swing video.
        """
        self.save_key_event_frames(video_path)
        self.plot_hand_path()
        self.plot_head_path()
        self.plot_feature_summary()
        self.plot_timing_events()