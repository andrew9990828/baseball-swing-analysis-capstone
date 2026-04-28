"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: pose_estimator.py
Description:
    Utilities for running pose estimation on video frames and
    standardizing landmark outputs for downstream swing analysis.

Last Updated: 4/27/26

Notes:
    Code implementation is being developed independently by Andrew Bieber.
    High-level scaffolding, planning, and pseudocode were discussed in collaboration.
"""

from dataclasses import dataclass

import cv2 as cv
import mediapipe as mp
import numpy as np


@dataclass
class PoseFrameResult:
    """
    Stores pose estimation output for a single frame.
    """

    frame_index: int
    landmarks: np.ndarray
    pose_detected: bool


class MediaPipePoseEstimator:
    """
    Wrapper around MediaPipe Pose for single-frame pose estimation
    and debug landmark drawing.
    """

    def __init__(self):
        """
        Initialize the MediaPipe Pose model and drawing utilities.
        """

        # self.mp_pose stores MediaPipe's pose module.
        self.mp_pose = mp.solutions.pose

        # self.mp_drawing stores MediaPipe's drawing utilities so we can draw skeleton landmarks later.
        self.mp_drawing = mp.solutions.drawing_utils

        # self.pose creates the pose estimator model.
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def estimate_frame(self, frame_bgr: np.ndarray, frame_index: int) -> PoseFrameResult:
        """
        Run pose estimation on a single BGR frame.

        Args:
            frame_bgr: Frame image in OpenCV BGR format.
            frame_index: Index of the frame in the original video.

        Returns:
            PoseFrameResult containing standardized landmark output.
        """

        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is None:
            # MediaPipe Pose gives 33 body landmarks.
            # We store 4 values per landmark: x, y, z, visibility.
            empty_landmarks = np.full((33, 4), np.nan, dtype=np.float32)

            return PoseFrameResult(
                frame_index=frame_index,
                landmarks=empty_landmarks,
                pose_detected=False,
            )

        landmarks = []

        for landmark in results.pose_landmarks.landmark:
            landmarks.append([
                landmark.x,
                landmark.y,
                landmark.z,
                landmark.visibility,
            ])

        landmark_array = np.array(landmarks, dtype=np.float32)

        return PoseFrameResult(
            frame_index=frame_index,
            landmarks=landmark_array,
            pose_detected=True,
        )

    def draw_landmarks(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Draw pose landmarks on a copy of the input frame.

        Args:
            frame_bgr: Frame image in OpenCV BGR format.

        Returns:
            Annotated frame with pose landmarks drawn.
        """

        output_frame = frame_bgr.copy()

        frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        if results.pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                output_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        return output_frame

    def close(self) -> None:
        """
        Release MediaPipe Pose resources.
        """

        self.pose.close()