# Module 7: Visualization / Report Assets

## Purpose

Module 7 creates visual proof assets for the baseball swing analysis pipeline.

The purpose of this module is to show the evidence behind the detected events, extracted features, and generated feedback.

Earlier modules answer:

```text
Module 4: What swing events were detected?
Module 5: What mechanics metrics were calculated?
Module 6: What feedback was generated from those metrics?
```

Module 7 answers:

```text
Can we visually prove and inspect those results?
```

This module does not add new baseball logic. It visualizes data that already exists from earlier modules.

---

## Current Module Role

Module 7 pulls from previous outputs:

```text
original video
cleaned landmark data
detected event frames
extracted feature values
```

Then it creates:

```text
key-frame images
trajectory plots
feature summary charts
event timeline charts
```

The main goal is visual debugging and proof, not polished final reporting yet.

---

## File Structure

```text
src/
  visualization/
    __init__.py
    frame_saver.py
    visualizer.py
    visualization_pipeline.py
    README.md
```

---

## Files

### `frame_saver.py`

Existing helper file from earlier modules.

Purpose:

```text
save selected video frames for debugging
```

This file can still be used for basic frame-saving tasks.

---

### `visualizer.py`

Contains the `SwingVisualizer` class.

This class owns the plotting and image-generation methods.

Current methods include:

```text
save_key_event_frames()
plot_hand_path()
plot_head_path()
plot_feature_summary()
plot_timing_events()
create_all_visualizations()
```

The visualizer receives:

```text
cleaned landmarks
detected events
extracted features
output directory
```

It creates the actual visualization files.

---

### `visualization_pipeline.py`

Connects the visualization module to previous module outputs.

This file loads:

```text
cleaned landmark data
detected event JSON
extracted feature JSON
original video path
```

Then it creates a `SwingVisualizer` object and writes the plots/images to disk.

Pipeline flow:

```text
load cleaned landmarks
load detected events
load extracted features
create SwingVisualizer
save key event frames
save trajectory plots
save feature summary chart
save timing event chart
```

---

## Current Inputs

```text
data/raw/mike_trout_swing_01.mp4
data/processed/pose/mike_trout_swing_01_pose_cleaned.npy
data/processed/events/mike_trout_swing_01_events.json
data/processed/features/mike_trout_swing_01_features.json
```

---

## Current Outputs

Current output directory:

```text
outputs/visualizations/mike_trout_swing_01/
```

Expected output files:

```text
key_frame_movement_start.jpg
key_frame_peak_hand_speed.jpg
key_frame_contact_proxy.jpg
hand_path_plot.png
head_path_plot.png
feature_summary.png
timing_events.png
```

---

## Current Visualizations

### Key Event Frames

Files:

```text
key_frame_movement_start.jpg
key_frame_peak_hand_speed.jpg
key_frame_contact_proxy.jpg
```

Purpose:

```text
Show the actual video frames where the system detected important swing events.
```

Current event frames:

```text
movement_start = frame 3
peak_hand_speed = frame 18
contact_proxy = frame 21
```

These images are useful because they allow visual inspection of whether Module 4 detected reasonable event positions.

---

### Timing Event Timeline

File:

```text
timing_events.png
```

Purpose:

```text
Show detected swing events on a simple frame-index timeline.
```

This makes it easy to see the spacing between:

```text
movement start
peak hand speed
contact proxy
```

For the Mike Trout sample, the detected timeline is:

```text
movement_start → frame 3
peak_hand_speed → frame 18
contact_proxy → frame 21
```

This supports the timing features calculated in Module 5.

---

### Head Movement Trajectory

File:

```text
head_path_plot.png
```

Purpose:

```text
Plot the nose/head landmark trajectory across frames.
```

This visual supports the `head_movement_start_to_contact` feature.

Important note:

The current plot uses the cleaned landmark data and may include more frames than just the main swing window. Because MediaPipe landmarks can jitter slightly, especially during fast motion or blur, the path may look noisy even when the final measured movement is small.

For v1, this plot is mainly a debug visual.

Future improvement:

```text
plot only the movement_start → contact_proxy window
```

This would make the chart more directly tied to the feature value.

---

### Hand Path Trajectory

File:

```text
hand_path_plot.png
```

Purpose:

```text
Plot the left and right wrist landmark paths across frames.
```

This visual supports the `hand_path_start_to_contact` feature.

The hand path is expected to show more motion than the head path because the hands move aggressively through the swing while the head should stay more controlled.

Important note:

The current plot may show extra motion or noise because it can include frames outside the main swing window.

Future improvement:

```text
plot only the movement_start → contact_proxy window
```

This would make the hand path visual easier to interpret.

---

### Feature Summary Chart

File:

```text
feature_summary.png
```

Purpose:

```text
Show selected Module 5 feature values in one quick debug chart.
```

Current plotted features:

```text
Head Movement
Hand Path
Shoulder Angle
Start to Contact
```

Important limitation:

This chart mixes different units:

```text
Head Movement = normalized landmark units
Hand Path = normalized landmark units
Shoulder Angle = degrees
Start to Contact = frames
```

Because the units are different, the bars should not be compared directly as if they are the same type of measurement.

For example, shoulder angle may dominate the chart because a value like `48 degrees` is numerically much larger than a normalized movement value like `0.35`.

This chart is useful as a quick debug summary, but it is not a clean scientific comparison chart yet.

Future improvement:

```text
distance_metrics.png
angle_metrics.png
timing_metrics.png
```

This would separate metrics by unit type.

---

## Important Visualization Notes

### Landmark Noise

Some noise is expected in the trajectory plots.

Reasons:

```text
MediaPipe landmark jitter
fast swing motion
motion blur
occlusion
video quality
frame-to-frame pose estimation variation
```

This does not mean the module failed.

The key question is whether the visuals are directionally useful and whether the detected frames/features are believable.

---

### Debug Visuals vs Final Report Visuals

The current Module 7 outputs are debug/report assets.

They are not final polished report pages yet.

Current goal:

```text
prove the pipeline is working
inspect event frames
inspect feature behavior
create assets for later reporting
```

Final polished reporting can happen in Module 8.

---

## Design Decisions

The main design decision was to keep visualization separate from calculation.

The structure is:

```text
Module 5 = calculate features
Module 6 = interpret features
Module 7 = visualize proof
```

This keeps the project modular.

The visualization module should not change feature values or feedback logic. It should only display what previous modules produced.

---

## Why Python Is Enough

Python is enough for this module.

Current visualization tools:

```text
OpenCV = read video frames and draw labels
Matplotlib = create plots and charts
NumPy = access landmark arrays
JSON = load events and features
```

No R or external reporting tool is needed for v1.

The graphs are simple, and the main goal is proof/debugging.

---

## Current Test Result

Module 7 successfully generated:

```text
key event frame images
event timeline chart
head movement trajectory plot
hand path trajectory plot
feature summary chart
```

The key event frames were especially useful because they showed that the detected event frames were visually reasonable for the Mike Trout swing sample.

The trajectory plots showed some expected landmark noise, but still provided useful visual evidence for the extracted metrics.

The feature summary chart worked as a quick debug view, but revealed that mixed-unit charts should be improved later.

---

## Future Improvements

Possible future improvements:

```text
plot only movement_start → contact_proxy for trajectory charts
separate feature summary charts by unit type
draw pose landmarks on key event frames
draw wrist/head paths directly on video frames
create side-by-side frame comparison image
add feedback text to report images
save a simple HTML or Markdown report
create an annotated video clip
```

Most important near-term improvements:

```text
1. Restrict trajectory plots to the swing window.
2. Separate mixed-unit feature charts.
3. Add pose overlays to key event frames.
```

---

## Engineering Lesson

Module 7 showed why visual proof matters.

The earlier modules produced numbers, but the plots and key frames make those numbers easier to inspect.

This module also revealed useful limitations:

```text
trajectory plots can look noisy if too many frames are included
feature summary charts should not mix units
key event frames are very useful for validating event detection
```

This is why visualization belongs before final packaging.

It helps catch issues before the final report/demo is created.

---

## Status

**Completed:** 5/25/26

Module 7 is complete enough for v1.

Current status:

```text
previous module outputs
→ visualization pipeline
→ saved proof assets
```

Module 7 now creates visual evidence for the swing analysis system.

The next step is Module 8: final end-to-end demo / report packaging.