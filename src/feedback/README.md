# Module 6: Feedback Engine

## Purpose

Module 6 turns extracted swing metrics into evidence-backed feedback.

Module 5 answers:

```text
What are the swing numbers?
```

Module 6 answers:

```text
What do those numbers suggest?
```

This module does not calculate new motion features. It interprets the feature dictionary produced by Module 5 using simple v1 rule-based logic.

---

## Current Inputs

Module 6 receives the feature dictionary from Module 5.

Current feature input example:

```text
head_movement_start_to_contact: 0.023977333679795265
hand_path_start_to_contact: 0.35641882877098396
hip_drift_start_to_contact: 1.4901161193847656e-08
shoulder_angle_change_start_to_contact: 48.16871643066406
frames_start_to_contact: 18
frames_start_to_peak_hand_speed: 15
frames_peak_hand_speed_to_contact: 3
```

These features come from:

```text
data/processed/features/mike_trout_swing_01_features.json
```

---

## Current Outputs

Module 6 produces a feedback dictionary containing:

```text
summary
feedback statements
warnings
numeric evidence
confidence levels
```

Example output structure:

```json
{
    "summary": "Swing metrics are mostly within the expected v1 ranges based on the current rule-based feedback engine.",
    "feedback": [
        {
            "metric": "head_movement_start_to_contact",
            "status": "good",
            "statement": "Head movement stayed controlled from movement start to contact proxy.",
            "evidence": "Measured value: 0.0240 normalized units. V1 good threshold: < 0.05.",
            "confidence": "medium"
        }
    ],
    "warnings": [
        {
            "metric": "hip_drift_start_to_contact",
            "warning": "Hip drift is not evaluated in Module 6 v1 because the current landmark data was normalized around the hip center in Module 3."
        }
    ]
}
```

---

## File Structure

```text
src/
  feedback/
    __init__.py
    feedback_engine.py
    README.md
```

---

## What was built

### `feedback_engine.py`

Contains the `FeedbackEngine` class.

The class receives:

```text
features dictionary from Module 5
```

It generates:

```text
feedback statements
status labels
numeric evidence
warning flags
summary text
```

Current v1 evaluation methods:

```text
evaluate_head_stability()
evaluate_hand_path()
evaluate_shoulder_rotation()
evaluate_timing()
```

Helper methods:

```text
_add_feedback()
_add_warning()
add_known_limitations()
generate_summary()
generate_feedback()
```

---

## Current Feedback Rules

### Head Stability

Metric:

```text
head_movement_start_to_contact
```

Units:

```text
normalized landmark coordinate units
```

Current v1 thresholds:

```text
good: value < 0.05
warning: 0.05 <= value <= 0.10
issue: value > 0.10
```

Purpose:

```text
Check whether the head stayed relatively stable from movement start to contact proxy.
```

---

### Hand Path

Metric:

```text
hand_path_start_to_contact
```

Units:

```text
normalized landmark coordinate units
```

Current v1 thresholds:

```text
good: value < 0.45
warning: 0.45 <= value <= 0.65
issue: value > 0.65
```

Purpose:

```text
Check whether the hands took a reasonable movement path from movement start to contact proxy.
```

---

### Shoulder Rotation Proxy

Metric:

```text
shoulder_angle_change_start_to_contact
```

Units:

```text
degrees
```

Current v1 thresholds:

```text
good: 25 <= value <= 65
warning: 15 <= value < 25
warning: 65 < value <= 80
issue: value < 15 or value > 80
```

Purpose:

```text
Check whether shoulder angle change is within an expected v1 range for rotation into contact.
```

---

### Timing

Metric:

```text
frames_start_to_contact
```

Units:

```text
frames
```

Current v1 thresholds:

```text
good: 10 <= value <= 25
warning: 8 <= value < 10
warning: 25 < value <= 35
issue: value < 8 or value > 35
```

Purpose:

```text
Check whether the time from movement start to contact proxy is within a believable frame range.
```

---

## Known Limitations

### Hip Drift Is Not Evaluated Yet

Metric:

```text
hip_drift_start_to_contact
```

Hip drift is not currently evaluated by Module 6.

Reason:

```text
Module 3 normalized every frame around the hip center.
```

Because the hip center is used as the origin, hip movement becomes nearly zero by design.

This means hip drift cannot be trusted from the current normalized landmark file.

Future fix:

```text
Save both cleaned non-normalized landmarks and normalized landmarks.
```

Then use:

```text
normalized landmarks → body-relative mechanics
non-normalized cleaned landmarks → body translation / drift metrics
```

---

### Thresholds Are Temporary

The current thresholds are hardcoded v1 placeholders.

They are based on:

```text
early testing
baseball reasoning
the current Mike Trout sample
directional expectations
```

These thresholds are not final scientific truth.

They should later be improved using more swing samples.

---

## Why Rule-Based Logic Makes Sense for V1

For v1, hardcoded if-else logic is appropriate because the goal is to prove that extracted metrics can be turned into readable feedback.

The current project flow is:

```text
video
→ pose landmarks
→ cleaned landmarks
→ detected events
→ extracted features
→ feedback statements
```

Module 6 does not need machine learning yet.

The purpose is to create a simple explainable feedback layer first.

---

## Future Improvement Path

The long-term plan is:

```text
V1: hardcoded if-else thresholds
V2: thresholds adjusted from more swing samples
V3: data-informed dynamic thresholds
V4: ML-assisted swing evaluation
```

New side-view swing clips from real college players can eventually be used as a test set or validation set.

The same pipeline can be run on each swing to build a feature table:

```text
player
head_movement
hand_path
shoulder_angle_change
frames_start_to_contact
feedback status
```

That data can later help determine better thresholds or train a model to recognize ideal and non-ideal swing patterns.

---

## Engineering Lesson

Module 6 is intentionally simple.

The important part is not complexity. The important part is the separation of responsibilities:

```text
Module 5 = calculate the numbers
Module 6 = interpret the numbers
```

This keeps the project modular and easier to improve.

The current if-else logic may be simple, but it creates a readable feedback system that can later be replaced, tuned, or expanded without changing the earlier modules.

---

## Status

Module 6 is complete enough for v1 testing.

Current status:

```text
Module 5 feature dictionary
→ Module 6 rule-based feedback
→ summary, feedback statements, evidence, and warnings
```

Next step:

```text
Run Module 6 on the Mike Trout feature output and verify that the feedback matches the visual swing and the extracted metrics.
```