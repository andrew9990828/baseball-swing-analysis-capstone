# Module 8: Evaluation

## Purpose

Module 8 verifies whether the v1 baseball swing analysis pipeline produces results that are stable, believable, and honestly documented.

This module does not add new swing mechanics calculations.

Instead, it evaluates the outputs from previous modules and creates a simple evaluation report.

For v1, this is a single-sample evaluation using the current Mike Trout swing video.

---

## Current Role

Earlier modules produce the actual analysis:

```text
Module 4 = detected swing events
Module 5 = extracted swing features
Module 6 = feedback statements
Module 7 = visualization assets
```

Module 8 checks and documents:

```text
Did the events happen in a believable order?
Did the features exist and make directional sense?
Did the feedback output have the expected structure?
Were the visualization files created?
What limitations still exist?
```

---

## File Structure

```text
src/
  evaluation/
    __init__.py
    evaluator.py
    README.md
```

---

## Files

### `evaluator.py`

Contains the `SwingEvaluator` class.

This class receives:

```text
sample name
detected events
extracted features
feedback output
visualization output folder
evaluation report output path
```

It then runs simple v1 checks and writes a markdown evaluation report.

Current methods include:

```text
check_event_order()
check_feature_values()
check_feedback_output()
check_visualization_outputs()
add_known_limitations()
run_evaluation_checks()
build_markdown_report()
save_report()
run()
```

---

## Current Inputs

Module 8 uses outputs from earlier modules:

```text
data/processed/events/mike_trout_swing_01_events.json
data/processed/features/mike_trout_swing_01_features.json
outputs/visualizations/mike_trout_swing_01/
```

It also receives the Module 6 feedback output directly from the current run.

---

## Current Output

Current report output:

```text
outputs/report/mike_trout_swing_01_evaluation.md
```

This report includes:

```text
pipeline status
detected events
extracted features
feedback summary
feedback items
evaluation notes
failure / review notes
known limitations
evaluation conclusion
```

---

## Evaluation Scope

This is not a full statistical validation.

The project currently only has one active test video:

```text
mike_trout_swing_01.mp4
```

So Module 8 evaluates whether the pipeline works end-to-end on that one sample and whether the outputs are believable enough for v1.

This is better described as:

```text
single-sample pipeline evaluation
```

not:

```text
full model validation
```

---

## Current Evaluation Checks

### Event Order Check

Checks whether the detected events occur in this order:

```text
movement start → peak hand speed → contact proxy
```

This is a basic sanity check for Module 4.

---

### Feature Value Check

Checks whether required v1 features exist:

```text
head_movement_start_to_contact
hand_path_start_to_contact
hip_drift_start_to_contact
shoulder_angle_change_start_to_contact
frames_start_to_contact
frames_start_to_peak_hand_speed
frames_peak_hand_speed_to_contact
```

Then it checks whether some values are directionally believable.

Examples:

```text
head movement should usually be lower than hand path distance
shoulder angle should be within a broad believable range
movement start to contact should be within a broad believable frame range
```

---

### Feedback Output Check

Checks whether Module 6 produced:

```text
summary
feedback
warnings
```

This verifies that the feedback engine returned the expected structure.

---

### Visualization Output Check

Checks whether Module 7 created the expected visual assets:

```text
key_frame_movement_start.jpg
key_frame_peak_hand_speed.jpg
key_frame_contact_proxy.jpg
hand_path_plot.png
head_path_plot.png
feature_summary.png
timing_events.png
```

This verifies that visual proof files were created.

---

## Known Limitations

The current v1 evaluation documents several limitations:

```text
only one sample video is currently evaluated
contact is a proxy because bat/ball contact is not tracked
distance metrics are normalized coordinate units, not real-world units
hip drift is unreliable because landmarks were normalized around the hip center
feedback thresholds are hardcoded placeholders
MediaPipe landmarks can jitter during fast motion or blur
```

These limitations are important because they prevent the system from being oversold.

---

## Why Module 8 Matters

Module 8 is about honesty.

The system now runs end-to-end, but that does not automatically mean every result is scientifically proven.

This module documents:

```text
what worked
what looked believable
what failed or needs review
what assumptions were made
what needs more data later
```

That makes the project stronger because the evaluation is not pretending the v1 system is final.

---

## Future Evaluation Plan

After v1, the evaluation should expand using more side-view swing videos.

Future evaluation steps:

```text
run the full pipeline on more swing samples
save feature JSON for each swing
compare feature distributions
track failure cases
record when event detection fails
record when pose landmarks are noisy
adjust hardcoded thresholds using data
create a validation set from real swing clips
eventually train ML models on extracted features
```

The college side-view swing clips can become the first post-v1 validation set.

---

## Status

**Completed:** 5/25/26

Module 8 is complete enough for v1.

Current status:

```text
previous module outputs
→ evaluator
→ markdown evaluation report
```

The project now has a complete v1 pipeline:

```text
video
→ pose
→ cleaned landmarks
→ events
→ features
→ feedback
→ visualization
→ evaluation report
```