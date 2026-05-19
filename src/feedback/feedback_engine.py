"""
Author: Andrew Bieber <andrewbieber.work@gmail.com>
File: feedback_engine.py
Description:
    Defines the swing feedback logic for Module 6.
    This module uses extracted swing features from Module 5 and applies
    simple v1 rule-based thresholds to generate evidence-backed feedback.

Last Updated: 5/19/26

Notes:
    This is a v1 feedback engine. The current thresholds are hardcoded
    placeholders based on early testing, baseball reasoning, and the current
    Mike Trout sample output. These thresholds should later be refined using
    a larger dataset of swing feature outputs.
"""

from typing import Dict, Any, List


class FeedbackEngine:
    """
    Generates swing feedback from extracted swing features.

    Module 6 does not calculate new motion features.
    It interprets the feature dictionary produced by Module 5.
    """

    def __init__(self, features: Dict[str, Any]):
        """
        Initialize the feedback engine.

        Args:
            features:
                Dictionary of extracted swing features from Module 5.
        """
        self.features = features
        self.feedback: List[Dict[str, Any]] = []
        self.warnings: List[Dict[str, Any]] = []

    def _add_feedback(
        self,
        metric: str,
        status: str,
        statement: str,
        evidence: str,
        confidence: str,
    ) -> None:
        """
        Add one feedback statement to the feedback list.

        Args:
            metric:
                Feature name being evaluated.

            status:
                Result of the rule check.
                Example: good, warning, issue, not_evaluated.

            statement:
                Human-readable feedback statement.

            evidence:
                Numeric evidence supporting the feedback.

            confidence:
                Confidence level for the feedback.
                Example: low, medium, high.
        """
        self.feedback.append(
            {
                "metric": metric,
                "status": status,
                "statement": statement,
                "evidence": evidence,
                "confidence": confidence,
            }
        )

    def _add_warning(self, metric: str, warning: str) -> None:
        """
        Add a warning message for metrics that should not be trusted yet.

        Args:
            metric:
                Feature name related to the warning.

            warning:
                Explanation of the limitation.
        """
        self.warnings.append(
            {
                "metric": metric,
                "warning": warning,
            }
        )

    def evaluate_head_stability(self) -> None:
        """
        Evaluate head stability from movement start to contact proxy.

        Current metric:
            head_movement_start_to_contact

        Units:
            normalized landmark coordinate units

        V1 threshold idea:
            good: value < 0.05
            warning: 0.05 <= value <= 0.10
            issue: value > 0.10
        """
        metric = "head_movement_start_to_contact"
        value = float(self.features[metric])

        if value < 0.05:
            status = "good"
            statement = (
                "Head movement stayed controlled from movement start "
                "to contact proxy."
            )
            evidence = (
                f"Measured value: {value:.4f} normalized units. "
                "V1 good threshold: < 0.05."
            )
        elif value <= 0.10:
            status = "warning"
            statement = (
                "Head movement was slightly elevated from movement start "
                "to contact proxy."
            )
            evidence = (
                f"Measured value: {value:.4f} normalized units. "
                "V1 warning range: 0.05 to 0.10."
            )
        else:
            status = "issue"
            statement = (
                "Head movement may be too high before contact. "
                "This could make it harder to stay stable through the swing."
            )
            evidence = (
                f"Measured value: {value:.4f} normalized units. "
                "V1 issue threshold: > 0.10."
            )

        self._add_feedback(
            metric=metric,
            status=status,
            statement=statement,
            evidence=evidence,
            confidence="medium",
        )

    def evaluate_hand_path(self) -> None:
        """
        Evaluate hand path distance from movement start to contact proxy.

        Current metric:
            hand_path_start_to_contact

        Units:
            normalized landmark coordinate units

        V1 threshold idea:
            good: value < 0.45
            warning: 0.45 <= value <= 0.65
            issue: value > 0.65
        """
        metric = "hand_path_start_to_contact"
        value = float(self.features[metric])

        if value < 0.45:
            status = "good"
            statement = (
                "Hand path distance stayed within the expected v1 range "
                "from movement start to contact proxy."
            )
            evidence = (
                f"Measured value: {value:.4f} normalized units. "
                "V1 good threshold: < 0.45."
            )
        elif value <= 0.65:
            status = "warning"
            statement = (
                "Hand path distance was moderately high. "
                "This may suggest a longer move to contact."
            )
            evidence = (
                f"Measured value: {value:.4f} normalized units. "
                "V1 warning range: 0.45 to 0.65."
            )
        else:
            status = "issue"
            statement = (
                "Hand path distance was high from movement start to contact proxy. "
                "This may suggest the hands are taking a long path to the ball."
            )
            evidence = (
                f"Measured value: {value:.4f} normalized units. "
                "V1 issue threshold: > 0.65."
            )

        self._add_feedback(
            metric=metric,
            status=status,
            statement=statement,
            evidence=evidence,
            confidence="medium",
        )

    def evaluate_shoulder_rotation(self) -> None:
        """
        Evaluate shoulder angle change from movement start to contact proxy.

        Current metric:
            shoulder_angle_change_start_to_contact

        Units:
            degrees

        V1 threshold idea:
            good: 25 <= value <= 65
            warning: 15 <= value < 25 or 65 < value <= 80
            issue: value < 15 or value > 80
        """
        metric = "shoulder_angle_change_start_to_contact"
        value = float(self.features[metric])

        if 25.0 <= value <= 65.0:
            status = "good"
            statement = (
                "Shoulder angle change was within the expected v1 range "
                "for rotation into contact."
            )
            evidence = (
                f"Measured value: {value:.2f} degrees. "
                "V1 good range: 25 to 65 degrees."
            )
        elif 15.0 <= value < 25.0:
            status = "warning"
            statement = (
                "Shoulder angle change was slightly low. "
                "This may suggest limited upper-body rotation into contact."
            )
            evidence = (
                f"Measured value: {value:.2f} degrees. "
                "V1 low warning range: 15 to 25 degrees."
            )
        elif 65.0 < value <= 80.0:
            status = "warning"
            statement = (
                "Shoulder angle change was slightly high. "
                "This may suggest aggressive rotation before contact."
            )
            evidence = (
                f"Measured value: {value:.2f} degrees. "
                "V1 high warning range: 65 to 80 degrees."
            )
        else:
            status = "issue"
            statement = (
                "Shoulder angle change was outside the expected v1 range. "
                "This could indicate limited rotation or excessive rotation."
            )
            evidence = (
                f"Measured value: {value:.2f} degrees. "
                "V1 issue range: < 15 or > 80 degrees."
            )

        self._add_feedback(
            metric=metric,
            status=status,
            statement=statement,
            evidence=evidence,
            confidence="medium",
        )

    def evaluate_timing(self) -> None:
        """
        Evaluate timing from movement start to contact proxy.

        Current metric:
            frames_start_to_contact

        Units:
            frames

        V1 threshold idea:
            good: 10 <= value <= 25
            warning: 8 <= value < 10 or 25 < value <= 35
            issue: value < 8 or value > 35
        """
        metric = "frames_start_to_contact"
        value = int(self.features[metric])

        if 10 <= value <= 25:
            status = "good"
            statement = (
                "The time from movement start to contact proxy was within "
                "the expected v1 frame range."
            )
            evidence = (
                f"Measured value: {value} frames. "
                "V1 good range: 10 to 25 frames."
            )
        elif 8 <= value < 10:
            status = "warning"
            statement = (
                "The move from movement start to contact proxy was very quick. "
                "This may need visual review."
            )
            evidence = (
                f"Measured value: {value} frames. "
                "V1 low warning range: 8 to 10 frames."
            )
        elif 25 < value <= 35:
            status = "warning"
            statement = (
                "The move from movement start to contact proxy was slightly long. "
                "This may suggest slower swing timing or a delayed contact proxy."
            )
            evidence = (
                f"Measured value: {value} frames. "
                "V1 high warning range: 25 to 35 frames."
            )
        else:
            status = "issue"
            statement = (
                "The timing from movement start to contact proxy was outside "
                "the expected v1 range."
            )
            evidence = (
                f"Measured value: {value} frames. "
                "V1 issue range: < 8 or > 35 frames."
            )

        self._add_feedback(
            metric=metric,
            status=status,
            statement=statement,
            evidence=evidence,
            confidence="medium",
        )

    def add_known_limitations(self) -> None:
        """
        Add known v1 warnings that should be shown with the feedback.
        """
        self._add_warning(
            metric="hip_drift_start_to_contact",
            warning=(
                "Hip drift is not evaluated in Module 6 v1 because the current "
                "landmark data was normalized around the hip center in Module 3. "
                "This makes hip center movement nearly zero by design."
            ),
        )

        self._add_warning(
            metric="thresholds",
            warning=(
                "Current feedback thresholds are hardcoded v1 placeholders. "
                "They should later be refined using more swing samples and "
                "feature distributions."
            ),
        )

    def generate_summary(self) -> str:
        """
        Generate a short summary based on feedback statuses.

        Returns:
            Summary string for the full swing feedback output.
        """
        issue_count = 0
        warning_count = 0
        good_count = 0

        for item in self.feedback:
            if item["status"] == "issue":
                issue_count += 1
            elif item["status"] == "warning":
                warning_count += 1
            elif item["status"] == "good":
                good_count += 1

        if issue_count > 0:
            return (
                f"Swing feedback found {issue_count} issue(s), "
                f"{warning_count} warning(s), and {good_count} good metric(s)."
            )

        if warning_count > 0:
            return (
                f"Swing feedback found {warning_count} warning(s) "
                f"and {good_count} good metric(s)."
            )

        return (
            "Swing metrics are mostly within the expected v1 ranges based on "
            "the current rule-based feedback engine."
        )

    def generate_feedback(self) -> Dict[str, Any]:
        """
        Run all v1 feedback checks.

        Returns:
            Dictionary containing summary, feedback statements, and warnings.
        """
        self.evaluate_head_stability()
        self.evaluate_hand_path()
        self.evaluate_shoulder_rotation()
        self.evaluate_timing()
        self.add_known_limitations()

        return {
            "summary": self.generate_summary(),
            "feedback": self.feedback,
            "warnings": self.warnings,
        }