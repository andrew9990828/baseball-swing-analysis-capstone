# Baseball Swing Analysis Capstone Plan

**Project start date:** 2026-04-22  
**Owner:** Andrew Bieber  
**Project type:** Explainable motion analysis / computer vision / applied AI pipeline  
**Current status:** V2 baseline and comparison pipeline complete  
**Primary lesson:** Modularity makes pipelines expandable; data quality determines whether downstream metrics are meaningful.

---

## 1. Project Summary

### One-sentence definition

Build an explainable baseball swing analysis pipeline that takes recorded swing videos, extracts pose-based motion data, measures interpretable mechanics, and generates evidence-backed reports.

### Project description

This project analyzes side-view baseball swing videos using a staged computer vision pipeline. The system extracts body pose landmarks from video, smooths and normalizes the motion data, detects rough swing events, calculates interpretable swing metrics, generates rule-based feedback, creates visual proof assets, and compares new swings against a small professional baseline.

The goal is not to claim full biomechanics accuracy. The goal is to build an honest, modular, inspectable system that turns raw video into measurable outputs while documenting assumptions and limitations.

---

## 2. Core Engineering Principles

- Build the smallest working version first.
- Prefer measurable outputs over vague AI claims.
- Keep each module responsible for one job.
- Treat data quality as a first-class part of the system.
- Use AI where it creates leverage, not everywhere.
- No feedback claim should exist without numeric or visual evidence.
- Finished and narrow beats ambitious and half-built.

---

## 3. Current System Overview

### V1 pipeline

```text
raw video
→ pose extraction
→ landmark processing
→ event detection
→ feature extraction
→ feedback generation
→ visualization output
→ evaluation report
```

### V2 pipeline

```text
folder of pro swing videos
→ batch processing through full V1 pipeline
→ aggregate feature table
→ professional baseline JSON
→ aggregate metric plots
→ unseen test swing processing
→ comparison report against professional baseline
```

---

## 4. Scope

### In scope

- Prerecorded swing videos
- One hitter in frame
- Side-view or near-side-view camera angle
- Pose extraction using MediaPipe
- OpenCV video ingestion
- NumPy-based feature calculations
- Rule-based event detection
- Rule-based feedback generation
- Median/IQR professional baseline
- Markdown/JSON reports
- Static plots and key-frame visualizations

### Out of scope for current version

- Real-time analysis
- Mobile app
- Multiple synchronized camera angles
- Custom pose model training
- Bat tracking
- Ball tracking
- Ground-truth biomechanics validation
- Automated drill recommendation
- Full coaching diagnosis
- ML scoring model without labeled data

---

## 5. Tech Stack

| Area | Tool |
|---|---|
| Language | Python |
| Video ingestion | OpenCV |
| Pose estimation | MediaPipe Pose |
| Numerical computation | NumPy |
| Plotting | Matplotlib |
| Data formats | JSON, CSV, NPY, NPZ, Markdown |
| Environment | Python virtual environment |

### Environment note

MediaPipe compatibility required pinning the environment carefully.

Working setup:

```text
Python 3.10.11
mediapipe==0.10.14
opencv-python
numpy
matplotlib
```

Earlier attempts using newer Python/MediaPipe combinations caused API compatibility issues. This reinforced that reproducibility depends on the whole stack, not just the code.

---

## 6. Repository Data Flow

### Raw inputs

```text
data/raw/pros/*.mp4
data/raw/tests/*.mp4
```

### Processed outputs

```text
data/processed/pose/*_pose_raw.npz
data/processed/pose/*_pose_cleaned.npy
data/processed/events/*_events.json
data/processed/features/*_features.json
```

### Report and visualization outputs

```text
outputs/debug_pose/
outputs/visualizations/
outputs/report/
outputs/aggregate/
outputs/comparisons/
```

### Important Git rule

Raw videos and large processed pose files should not be committed.

Keep these ignored:

```gitignore
data/raw/**/*.mp4
data/processed/pose/*.npy
data/processed/pose/*.npz
outputs/debug_pose/
outputs/visualizations/
```

Aggregate reports, plots, feature tables, and comparison markdown files are acceptable to commit because they show the project output without storing raw video data.

---

## 7. Module Breakdown

## Module 1: Video Ingestion

**Purpose:** Load and inspect raw video input.

### Responsibilities

- Load a video file
- Extract frames
- Track frame indices and timestamps
- Save debug frames for sanity checking
- Help identify bad input clips before downstream processing

### Key lesson

Video ingestion is not just file loading. Bad trimming, repeated frames, poor camera angle, or dead time can affect every later module.

### Status

Complete for V1.

---

## Module 2: Pose Extraction

**Purpose:** Run MediaPipe Pose on each frame and save standardized landmark data.

### Main files

```text
src/pose/pose_estimator.py
src/pose/pose_pipeline.py
```

### Responsibilities

- Open raw video with OpenCV
- Run pose estimation frame by frame
- Save landmark arrays
- Save detection flags
- Save pose debug frames

### Output format

```text
(num_frames, 33, 4)
```

Meaning:

```text
axis 0 = frame over time
axis 1 = MediaPipe landmark index
axis 2 = x, y, z, visibility
```

### Key lesson

The pose model is an AI dependency, but the project is the pipeline built around it. The system still needs preprocessing, data handling, validation, feature extraction, reporting, and honest limitations.

### Status

Complete for V1 and used in V2 batch processing.

---

## Module 3: Landmark Processing

**Purpose:** Clean and normalize raw pose landmarks.

### Main file

```text
src/processing/landmark_processor.py
```

### Responsibilities

- Load raw `.npz` pose output
- Extract landmark arrays
- Smooth landmark trajectories
- Normalize positions relative to hip center
- Save cleaned landmark `.npy` file

### Current design

The current cleaned landmark file is normalized around the hip center:

```text
normalized_landmarks = landmarks - hip_center
```

This makes body-relative motion easier to measure, but it also means whole-body hip drift cannot be measured from the normalized output.

### Key lesson

Different features may require different data representations.

```text
normalized landmarks → body-relative mechanics
non-normalized landmarks → whole-body translation / drift
```

### Status

Complete for V1. Future improvement: save both normalized and non-normalized cleaned landmark files.

---

## Module 4: Event Detection

**Purpose:** Identify rough swing event frames.

### Main files

```text
src/events/event_detector.py
src/events/event_pipeline.py
```

### Detected events

```text
movement_start
peak_hand_speed
contact_proxy
```

### Current method

- Uses wrist movement as the main swing signal
- Calculates a wrist-speed proxy
- Detects movement start when wrist speed passes a threshold
- Detects peak hand speed as the max wrist speed frame
- Estimates contact proxy a few frames after peak hand speed

### Important limitation

The system does not track the bat or ball, so contact is a proxy, not ground truth.

### Status

Complete for V1. Works for exploratory analysis but remains sensitive to clip trimming, slow motion, and non-swing movement.

---

## Module 5: Feature Extraction

**Purpose:** Convert cleaned motion data and detected events into interpretable swing metrics.

### Main files

```text
src/features/feature_extractor.py
src/features/feature_pipeline.py
```

### Current features

```text
head_movement_start_to_contact
hand_path_start_to_contact
hip_drift_start_to_contact
shoulder_angle_change_start_to_contact
frames_start_to_contact
frames_start_to_peak_hand_speed
frames_peak_hand_speed_to_contact
```

### Feature meanings

#### Head movement

Measures nose/head landmark displacement from movement start to contact proxy.

This is a pose-derived proxy for head stability.

#### Hand path distance

Measures average wrist-path distance from movement start to contact proxy.

This is a pose-derived proxy for hand path movement, not a full bat-path measurement.

#### Hip drift

Currently unreliable because landmarks are normalized around the hip center.

#### Shoulder angle change

Measures shoulder-line angle change between movement start and contact proxy.

Current issue: the angle calculation can produce wraparound artifacts, such as values near 300 degrees. This metric is reported as experimental/noisy.

#### Timing metrics

Frame differences between detected events.

Timing metrics are sensitive to clip frame rate, slow motion, trimming, and event detection behavior.

### Status

Complete for V1 and V2. Some metrics are core; some are experimental.

---

## Module 6: Feedback Engine

**Purpose:** Turn feature values into readable, evidence-backed feedback.

### Main file

```text
src/feedback/feedback_engine.py
```

### Responsibilities

- Apply simple rule-based checks
- Generate status labels
- Attach numeric evidence
- Include known limitation warnings
- Keep interpretation separate from feature calculation

### Current method

The feedback engine uses if/else logic and hardcoded thresholds.

This is intentional. The current dataset is too small and unlabeled for a meaningful ML model. Rule-based feedback is more transparent and appropriate at this stage.

### Key lesson

Not every part of an AI project needs ML. Pose estimation is the AI dependency; the downstream interpretation can be deterministic and explainable.

### Status

Complete for V1. Future thresholds can be refined using baseline statistics.

---

## Module 7: Visualization

**Purpose:** Create visual proof for detected events and metrics.

### Main files

```text
src/visualization/visualizer.py
src/visualization/visualization_pipeline.py
```

### Outputs

```text
key_frame_movement_start.jpg
key_frame_peak_hand_speed.jpg
key_frame_contact_proxy.jpg
hand_path_plot.png
head_path_plot.png
feature_summary.png
timing_events.png
```

### Key lesson

Visualizations are not just presentation assets. They are debugging tools that help validate whether the pipeline is telling the truth.

### Known issue

The feature summary chart mixes normalized units, degrees, and frame counts. It is useful for debugging but should not be treated as a scientific comparison chart.

### Status

Complete for V1. V2 aggregate plots separate movement, rotation, and timing metrics.

---

## Module 8: Evaluation

**Purpose:** Generate an honest evaluation report for a processed swing.

### Main file

```text
src/evaluation/evaluator.py
```

### Responsibilities

- Check event order
- Check required feature values
- Check feedback output structure
- Check visualization files
- Document known limitations
- Save markdown report

### Status

Complete for V1. Used per video in the batch pipeline.

---

## Module 9: Batch Professional Baseline

**Purpose:** Run the full pipeline across multiple professional swing clips and build a baseline.

### Main file

```text
src/main.py
```

### Responsibilities

- Loop through all `.mp4` files in a selected raw folder
- Run the full pipeline for each video
- Save per-video features, events, visualizations, and reports
- Aggregate all feature JSON files into a CSV table
- Compute professional baseline statistics
- Save aggregate plots

### Current professional sample

```text
data/raw/pros/*.mp4
```

Current baseline size:

```text
10 professional swing clips
```

### Baseline method

For each selected metric:

```text
median
q1
q3
iqr
min
max
count
```

Median/IQR are used because the sample size is small and outliers are likely.

### Output files

```text
outputs/aggregate/pro_feature_table.csv
outputs/aggregate/pro_baseline.json
outputs/aggregate/pro_batch_summary.json
outputs/aggregate/pro_movement_metrics.png
outputs/aggregate/pro_rotation_metrics.png
outputs/aggregate/pro_timing_metrics.png
```

### Status

Complete for V2.

---

## Module 10: Comparison Report

**Purpose:** Compare an unseen test swing against the professional baseline.

### Main file

```text
src/comparison/compare_metrics.py
```

### Responsibilities

- Load one feature JSON file
- Load `pro_baseline.json`
- Compare sample metrics against pro Q1/median/Q3
- Separate core metrics from experimental/noisy metrics
- Generate markdown and JSON comparison reports

### Example command

```powershell
python -m src.comparison.compare_metrics --features data/processed/features/college_swing_02_features.json --sample-name college_swing_02
```

### Output files

```text
outputs/comparisons/college_swing_02_vs_pro_baseline.md
outputs/comparisons/college_swing_02_vs_pro_baseline.json
```

### Status

Complete for V2.

---

## 8. Core vs Experimental Metrics

### Core metrics

These are useful enough for the current comparison report:

```text
head_movement_start_to_contact
hand_path_start_to_contact
frames_peak_hand_speed_to_contact
```

### Experimental / noisy metrics

These are reported for transparency but should not be treated as strong conclusions:

```text
shoulder_angle_change_start_to_contact
frames_start_to_contact
frames_start_to_peak_hand_speed
```

### Excluded or limited metric

```text
hip_drift_start_to_contact
```

Hip drift is currently limited because cleaned landmarks are normalized around the hip center.

---

## 9. Professional Baseline Notes

The professional baseline is not an ideal swing model.

It is a small sample of professional side-view clips processed through the same pipeline. Its purpose is to provide a rough comparison range, not a final definition of good mechanics.

### Current baseline report language

```text
The professional baseline is computed from 10 side-view professional swing clips using median and interquartile range. Metrics are pose-derived and intended for relative comparison, not ground-truth biomechanics. Some metrics are excluded or marked experimental when pose quality, camera angle, or event detection produces unreliable values.
```

---

## 10. Comparison Report Guardrails

Every comparison report should state:

- This is exploratory, not a coaching diagnosis.
- Values are pose-derived estimates, not ground-truth biomechanics.
- The professional baseline is not an ideal swing model.
- Camera angle, frame rate, clip trimming, and pose quality affect outputs.
- Contact is a proxy because bat and ball are not tracked.
- Experimental metrics should not be treated as strong conclusions.

---

## 11. Rule-Based vs ML-Based Split

### Current decision

The current version should remain:

```text
pose-based + NumPy feature engineering + rule-based interpretation + baseline statistics
```

### Why not add ML now?

A meaningful ML layer would require at least one of the following:

- Coach-labeled swing flaws
- Ground-truth event frames
- Bat speed or exit velocity labels
- Many more swing samples
- A clear target variable

Without labels, forcing ML would make the project weaker and less honest.

### Where ML may belong later

- Learned swing phase detection
- Swing pattern clustering
- Anomaly detection
- Coach-labeled issue classification
- Regression against bat speed or exit velocity

### Key lesson

Use AI where it provides leverage. Do not force ML into every part of the system.

---

## 12. Architecture Decisions

### Decision 1: One camera angle only

**Choice:** Side-view or near-side-view swings only.  
**Reason:** Different views create different geometry and make metrics inconsistent.

### Decision 2: Use an existing pose estimator

**Choice:** MediaPipe Pose.  
**Reason:** Pose estimation is a dependency, not the core capstone contribution.

### Decision 3: Interpretability first

**Choice:** Feature engineering and rule-based interpretation before ML scoring.  
**Reason:** Easier to debug, explain, and validate.

### Decision 4: Modular pipeline

**Choice:** Separate each stage into modules.  
**Reason:** V1 could become V2 without rewriting the entire project.

### Decision 5: Evidence-backed feedback only

**Choice:** No feedback without numeric or visual support.  
**Reason:** Prevents vague or fake coaching claims.

### Decision 6: Separate core and experimental metrics

**Choice:** Report noisy metrics transparently but avoid overclaiming.  
**Reason:** Some pose-derived metrics are useful for debugging but not yet reliable enough for conclusions.

---

## 13. Major Risks and Mitigations

### Risk 1: Pose noise

MediaPipe landmarks may jitter or fail during blur, occlusion, or fast motion.

**Mitigation:** Smooth landmarks, inspect debug frames, and treat noisy metrics as experimental.

### Risk 2: Camera inconsistency

Different camera angles change the meaning of normalized movement metrics.

**Mitigation:** Restrict current system to side-view clips.

### Risk 3: Event detection ambiguity

Movement start and contact proxy are heuristic, not ground truth.

**Mitigation:** Save key frames and event timelines for visual inspection.

### Risk 4: Clip trimming quality

Extra dead time, multiple swings, or slow-motion clips can distort timing and hand-path metrics.

**Mitigation:** Prefer one swing per clip from setup through follow-through.

### Risk 5: Overclaiming

It is easy to make the system sound more accurate than it is.

**Mitigation:** Use guardrails in reports and document limitations clearly.

---

## 14. Current Commands

### Run pro batch pipeline

```powershell
python -m src.main --raw-dir data/raw/pros --group-label pro
```

### Reuse existing intermediates when appropriate

```powershell
python -m src.main --raw-dir data/raw/pros --group-label pro --skip-existing
```

### Run test swing pipeline

```powershell
python -m src.main --raw-dir data/raw/tests --group-label test
```

### Compare test swing against pro baseline

```powershell
python -m src.comparison.compare_metrics --features data/processed/features/college_swing_02_features.json --sample-name college_swing_02
```

---

## 15. Definition of Success

### MVP success

The MVP is successful if it can:

- Process one swing video
- Extract pose trajectories
- Detect rough swing events
- Compute interpretable metrics
- Generate simple feedback
- Produce visual proof assets
- Save an evaluation report

### V2 success

The V2 system is successful if it can:

- Process multiple professional swing clips
- Aggregate feature outputs
- Build a professional median/IQR baseline
- Process an unseen test swing
- Compare the test swing against the pro baseline
- Clearly separate core metrics from experimental/noisy metrics
- Document limitations honestly

### Capstone success

The capstone is successful if it demonstrates:

- End-to-end pipeline design
- Modular software structure
- Applied computer vision
- Feature engineering
- Rule-based interpretation
- Data-quality awareness
- Technical documentation
- Honest evaluation and limitation handling

---

## 16. Build Log

### 2026-04-22

Project started.

Completed:

- Defined project direction
- Defined final vision vs MVP
- Created staged build roadmap
- Identified first technical milestone

### 2026-04-25

Module 1 completed.

Completed:

- Loaded raw video
- Extracted frames and timestamps
- Saved debug frames
- Identified that raw video quality and trimming affect downstream processing

### 2026-04-27

Module 2 completed.

Completed:

- Integrated MediaPipe Pose
- Extracted per-frame landmark arrays
- Saved raw pose data as `.npz`
- Saved debug pose frames
- Resolved Python/MediaPipe compatibility issue

### 2026-04-30

Module 3 completed.

Completed:

- Loaded raw pose data
- Smoothed landmarks
- Normalized landmarks around hip center
- Saved cleaned landmark `.npy`
- Identified hip drift limitation caused by normalization

### 2026-05-01

Module 4 completed.

Completed:

- Implemented wrist-speed event detection
- Detected movement start, peak hand speed, and contact proxy
- Saved event JSON outputs

### 2026-05-18

Modules 5 and 6 completed.

Completed:

- Extracted pose-derived swing features
- Built feedback engine using rule-based thresholds
- Added warnings for known limitations
- Confirmed simple if/else logic is appropriate for this stage

### 2026-05-25

Modules 7 and 8 completed.

Completed:

- Generated key-frame visualizations
- Generated movement and timing plots
- Created evaluation report
- Completed V1 end-to-end pipeline

### 2026-06-17

V2 baseline and comparison layer completed.

Completed:

- Processed 10 professional swing clips through full pipeline
- Generated aggregate professional feature table
- Created `pro_baseline.json`
- Created professional movement, rotation, and timing plots
- Added comparison module for unseen test swings
- Generated college swing vs professional baseline comparison report
- Confirmed core lesson: modularity enabled the V1 pipeline to become a V2 batch system quickly

---

## 17. Current Limitations

- Only side-view or near-side-view videos are supported.
- Contact is a proxy because bat and ball are not tracked.
- Distance metrics are normalized landmark units, not inches or centimeters.
- Hip drift is unreliable from normalized landmarks.
- Shoulder angle change can suffer from angle wraparound artifacts.
- Timing metrics are sensitive to clip speed, frame rate, trimming, and event detection.
- The professional baseline uses a small sample size.
- The comparison report is exploratory, not a coaching diagnosis.

---

## 18. Future Work

### Highest-value technical improvements

- Save both normalized and non-normalized cleaned landmarks.
- Fix shoulder angle wraparound.
- Add manual trim/start/end frame support.
- Plot trajectories only inside the detected swing window.
- Improve event detection using more robust multi-signal heuristics.
- Add per-video data-quality scoring.
- Add a manifest file for player metadata and handedness.
- Add a small smoke-test suite.
- Create a cleaner README with example outputs.

### Possible ML extensions later

Only after more data and labels:

- Swing phase detection from landmark trajectories
- Swing similarity clustering
- Anomaly detection
- Coach-labeled flaw classification
- Regression against bat speed or exit velocity

---

## 19. Final Guiding Reminder

This project should be described honestly:

```text
An explainable, pose-based computer vision pipeline for baseball swing analysis.
```

Not:

```text
A fully validated biomechanics system.
```

The strongest version of the project is not fake ML. It is a modular, evidence-backed pipeline that uses AI where appropriate, simple rules where appropriate, and documents its limitations clearly.
