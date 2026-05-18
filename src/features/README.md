# Module 5: Feature Extraction

## Purpose

Module 5 calculates interpretable swing mechanics metrics from cleaned pose landmark data and detected swing events.

This module connects:

```text
Module 3 cleaned landmarks
+ Module 4 detected swing events
→ Module 5 swing features
```

The goal is not to give coaching feedback yet. The goal is to produce useful measurements that can later be interpreted by a feedback module.

---

## Current Inputs

```text
data/processed/pose/mike_trout_swing_01_pose_cleaned.npy
data/processed/events/mike_trout_swing_01_events.json
```

The cleaned landmark file contains pose data with shape:

```text
(num_frames, 33, 4)
```

Meaning:

```text
axis 0 = frame over time
axis 1 = MediaPipe landmark index
axis 2 = x, y, z, visibility
```

The event JSON contains the rough swing events detected in Module 4:

```json
{
    "movement_start": 3,
    "peak_hand_speed": 18,
    "contact_proxy": 21
}
```

---

## Current Output

```text
data/processed/features/mike_trout_swing_01_features.json
```

Example output from the first Module 5 test run:

```text
head_movement_start_to_contact: 0.023977333679795265
hand_path_start_to_contact: 0.35641882877098396
hip_drift_start_to_contact: 1.4901161193847656e-08
shoulder_angle_change_start_to_contact: 48.16871643066406
frames_start_to_contact: 18
frames_start_to_peak_hand_speed: 15
frames_peak_hand_speed_to_contact: 3
```

---

## Current Feature Meanings

### Head Movement

```text
head_movement_start_to_contact
```

Measures how far the nose landmark moved from movement start to contact proxy.

This is currently measured in normalized landmark coordinate units, not inches or centimeters.

For the Mike Trout sample, the value was very small, which directionally makes sense for an elite hitter with a stable head through the swing.

---

### Hand Path Distance

```text
hand_path_start_to_contact
```

Measures the average wrist path distance from movement start to contact proxy.

This uses both wrist landmarks and averages their total path distance over time.

This value should normally be much larger than head movement because the hands are supposed to travel during the swing while the head should remain more stable.

---

### Hip Drift

```text
hip_drift_start_to_contact
```

Measures hip center movement from movement start to contact proxy.

Important note:

This value is currently not useful because Module 3 normalized every frame around the hip center. Since the hip center is used as the origin, hip drift becomes almost zero by design.

This is not a math bug. It is a data-design issue.

Design implication:

```text
normalized landmarks = useful for body-relative mechanics
raw / cleaned non-normalized landmarks = needed for whole-body movement features
```

Future improvement:

Save both versions of landmark data:

```text
cleaned_landmarks.npy
normalized_landmarks.npy
```

Then use:

```text
normalized landmarks → body-relative mechanics
cleaned non-normalized landmarks → body translation / drift metrics
```

---

### Shoulder Angle Change

```text
shoulder_angle_change_start_to_contact
```

Measures the angle change of the shoulder line from movement start to contact proxy.

This uses the line between the left shoulder and right shoulder landmarks.

Unlike the distance metrics, this output is in degrees.

For the first Mike Trout test, the shoulder angle changed about 48 degrees from movement start to contact proxy, which is directionally believable for a swing rotation metric.

---

### Timing Features

```text
frames_start_to_contact
frames_start_to_peak_hand_speed
frames_peak_hand_speed_to_contact
```

These values are measured in frames.

For the current test video:

```text
frames_start_to_contact = 18
frames_start_to_peak_hand_speed = 15
frames_peak_hand_speed_to_contact = 3
```

If the video is 30 FPS, then:

```text
18 frames / 30 FPS = 0.60 seconds
15 frames / 30 FPS = 0.50 seconds
3 frames / 30 FPS = 0.10 seconds
```

---

## Important Unit Clarification

The current distance-based outputs are not real-world units.

They are not:

```text
inches
centimeters
feet
meters
```

They are normalized pose-coordinate distances.

This means the current distance features are best used for relative comparison inside the same video or between videos processed in the same way.

Examples:

```text
head movement compared to hand path
movement from start to contact
relative stability of one body part versus another
```

They should not yet be interpreted as exact physical measurements.

---

## Early Observation From First Test

The first Module 5 run produced believable values for:

```text
head movement
hand path distance
shoulder angle change
timing relationships
```

Since the sample swing is Mike Trout, the low head movement and clean timing are directionally reasonable.

The biggest discovery was that hip drift became essentially zero because the cleaned landmark file had already been normalized around the hip center.

This clarified an important architecture decision:

```text
Some features need normalized data.
Some features need non-normalized cleaned data.
```

That means Module 5 may eventually need access to both versions of the landmark output.

---

## Engineering Lesson

Module 5 showed the value of modular design.

The current structure is:

```text
feature_extractor.py = reusable calculation object
feature_pipeline.py = connects files and runs the process
SwingFeatures dataclass = clean output container
NumPy arrays = compact motion data across frames
JSON output = readable downstream result
```

This made the project easier to understand, test, and improve.

The feature extractor owns the calculation logic.

The feature pipeline owns the file loading, object creation, feature extraction, and saving process.

This separation keeps the code clean and makes it easier to add or replace features later.

---

## Status

Module 5 is working as a v1 feature extraction system.

Current status:

```text
cleaned pose landmarks
+ detected swing events
→ extracted swing feature JSON
```

The next step is not to immediately overbuild the feature list.

The next step is to watch the video, compare the numbers to the actual swing, and decide which metrics are useful enough to keep.

Future modules can use these features to produce human-readable swing feedback.