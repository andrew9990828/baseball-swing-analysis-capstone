# Baseball Swing Analysis Capstone

An explainable computer vision pipeline for analyzing baseball swing videos with pose estimation, feature extraction, rule-based feedback, visual validation, and professional-baseline comparison.

This project takes side-view baseball swing clips, extracts pose landmarks over time, detects rough swing events, calculates interpretable motion metrics, generates visual outputs, and compares test swings against a small professional baseline.

The goal is not to create a black-box coaching model. The goal is to build a modular, inspectable motion-analysis pipeline where every metric and feedback statement is supported by numeric or visual evidence.

---

## Project Status

**Current status:** V2 complete enough for demonstration and resume use.

The project now supports:

- full video ingestion from `.mp4` files
- MediaPipe pose extraction
- landmark smoothing and normalization
- swing event detection
- feature extraction
- rule-based feedback
- visualization outputs
- markdown evaluation reports
- batch processing over multiple professional swings
- professional baseline generation using median/IQR statistics
- comparison reports for unseen test swings

---

## High-Level Pipeline

```text
raw swing video
    -> pose extraction
    -> landmark processing
    -> event detection
    -> feature extraction
    -> feedback generation
    -> visualization output
    -> evaluation report
    -> aggregate baseline statistics
    -> test swing comparison report
```

---

## Why This Project Exists

Baseball swing analysis is easy to overstate. Many tools can produce feedback that sounds intelligent, but is not clearly tied to measurable evidence.

This project takes the opposite approach:

- measure first
- interpret second
- show evidence
- document limitations
- avoid fake precision

A major lesson from building this project was that **data quality controls everything**. Camera angle, trimming, frame rate, pose visibility, and clip consistency all directly affect downstream metrics and feedback.

---

## Core Design Principles

### 1. Modularity

Each stage owns one job:

```text
video_loader.py          -> raw video metadata and frame extraction
pose_estimator.py        -> pose estimation on one frame
pose_pipeline.py         -> full-video pose extraction
landmark_processor.py    -> smoothing and normalization
event_detector.py        -> swing event detection
feature_extractor.py     -> metric calculation
feedback_engine.py       -> rule-based feedback
visualizer.py            -> plots and key-frame outputs
evaluator.py             -> markdown evaluation report
compare_metrics.py       -> test swing vs pro baseline comparison
main.py                  -> batch orchestration
```

This made the project much easier to extend from a one-video MVP into a batch-processing baseline system.

### 2. Explainability before ML

This project uses applied AI where it provides leverage: pose estimation.

The downstream system intentionally uses deterministic logic:

```text
pose estimation        -> applied AI dependency
feature extraction     -> NumPy/math
swing event detection  -> heuristics
feedback generation    -> if/else rules
baseline comparison    -> median/IQR statistics
```

Machine learning does not belong in every layer. For this stage of the project, simple rules and statistical baselines are more honest and more inspectable than forcing a model without enough labeled data.

### 3. Evidence-backed feedback

The system should not produce swing feedback unless it can point to supporting evidence:

- detected event frames
- extracted metrics
- plots
- comparison tables
- warnings and limitations

---

## Current Metrics

### Core Metrics

These are the most useful metrics in the current pipeline:

| Metric | Meaning | Unit |
|---|---|---|
| `head_movement_start_to_contact` | Nose/head displacement from movement start to contact proxy | normalized units |
| `hand_path_start_to_contact` | Average wrist path distance from movement start to contact proxy | normalized units |
| `frames_peak_hand_speed_to_contact` | Frames between peak wrist speed and contact proxy | frames |

### Experimental / Noisy Metrics

These are currently reported, but should be treated carefully:

| Metric | Reason for caution |
|---|---|
| `shoulder_angle_change_start_to_contact` | Can suffer from angle wraparound artifacts |
| `frames_start_to_contact` | Sensitive to trimming, frame rate, and event detection behavior |
| `frames_start_to_peak_hand_speed` | Sensitive to trimming, frame rate, and event detection behavior |
| `hip_drift_start_to_contact` | Not reliable because cleaned landmarks are normalized around hip center |

---

## Professional Baseline

The V2 pipeline processes professional side-view swing clips and aggregates feature values into a professional baseline.

Current baseline method:

```text
for each metric:
    collect valid finite values across professional swings
    compute median
    compute Q1
    compute Q3
    compute IQR
    save results to JSON
```

Median and IQR are used instead of mean and standard deviation because the sample size is small and the video sources vary in quality.

The professional baseline is **not** an ideal swing model. It is a sample-based reference built from pose-derived metrics.

Output:

```text
outputs/aggregate/pro_baseline.json
outputs/aggregate/pro_feature_table.csv
outputs/aggregate/pro_movement_metrics.png
outputs/aggregate/pro_rotation_metrics.png
outputs/aggregate/pro_timing_metrics.png
```

---

## Test Swing Comparison

A test swing can be processed through the same pipeline and compared against the professional baseline.

Example output:

```text
outputs/comparisons/college_swing_02_vs_pro_baseline.md
outputs/comparisons/college_swing_02_vs_pro_baseline.json
```

The comparison report separates metrics into:

- core metrics
- experimental/noisy metrics
- interpretation guardrails

The report is intended for exploratory analysis, not final coaching advice.

---

## Repository Structure

```text
baseball-swing-analysis-capstone/

├── data/
│   ├── raw/
│   │   ├── pros/
│   │   └── tests/
│   └── processed/
│       ├── events/
│       ├── features/
│       └── pose/

├── outputs/
│   ├── aggregate/
│   ├── comparisons/
│   ├── debug_frames/
│   ├── debug_pose/
│   ├── report/
│   └── visualizations/

├── src/
│   ├── comparison/
│   ├── evaluation/
│   ├── events/
│   ├── features/
│   ├── feedback/
│   ├── io/
│   ├── pose/
│   ├── processing/
│   └── visualization/

├── tests/
├── notebooks/
├── PLAN.md
├── README.md
└── requirements.txt
```

---

## Setup

### 1. Create a virtual environment

```powershell
python -m venv .venv
```

### 2. Activate the virtual environment

PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

### 3. Install dependencies

```powershell
pip install -r requirements.txt
```

Important environment note:

```text
Python 3.10.11
MediaPipe 0.10.14
```

MediaPipe compatibility was a real project constraint. Python/package version mismatches can break the pose-estimation layer.

---

## Running the Pipeline

### Process professional swings and build baseline

Place professional swing clips in:

```text
data/raw/pros/
```

Then run:

```powershell
python -m src.main --raw-dir data/raw/pros --group-label pro
```

This generates processed pose data, feature JSON files, visualizations, evaluation reports, and aggregate baseline outputs.

### Process test swings

Place test swing clips in:

```text
data/raw/tests/
```

Then run:

```powershell
python -m src.main --raw-dir data/raw/tests --group-label test
```

### Compare a test swing against the pro baseline

Example:

```powershell
python -m src.comparison.compare_metrics --features data/processed/features/college_swing_02_features.json --sample-name college_swing_02
```

This creates:

```text
outputs/comparisons/college_swing_02_vs_pro_baseline.md
outputs/comparisons/college_swing_02_vs_pro_baseline.json
```

---

## Example Comparison Output

Example core metric table:

```text
| Metric | Sample Value | Pro Q1 | Pro Median | Pro Q3 | Status |
|---|---:|---:|---:|---:|---|
| Head movement | 0.0575 | 0.0287 | 0.0438 | 0.0652 | within_pro_middle_50 |
| Hand path distance | 0.2576 | 0.5216 | 1.0801 | 1.0929 | below_pro_middle_50 |
| Peak hand speed to contact | 3.0000 | 3.0000 | 3.0000 | 3.0000 | within_pro_middle_50 |
```

The comparison output is useful because it shows the pipeline can process an unseen swing and compare it against the pro sample distribution. It should not be treated as a definitive swing-quality diagnosis.

---

## Limitations

Current limitations:

- only side-view swing videos are supported
- camera angle consistency matters
- MediaPipe pose landmarks can jitter during fast motion or occlusion
- bat and ball are not tracked
- contact is a proxy, not true contact detection
- distance values are normalized landmark units, not inches or centimeters
- hip drift is unreliable because the cleaned landmarks are normalized around hip center
- timing metrics are sensitive to clip trimming, frame rate, and slow-motion footage
- shoulder angle change can suffer from angle wraparound artifacts
- the professional baseline is small and exploratory
- comparison reports are not coaching diagnoses

These limitations are part of the project, not hidden failures. The pipeline is designed to make failure modes visible through reports, plots, and evaluation notes.

---

## Key Lessons Learned

### Data quality controls the pipeline

Poor trimming, inconsistent angles, slow-motion footage, repeated frames, and missing landmarks can distort downstream metrics.

### Modularity matters

The project became easier to extend because each module owned one responsibility. The batch V2 system could reuse the original V1 modules instead of requiring a full rewrite.

### Simple logic is often enough

Not every part of a project needs machine learning. For this use case, pose estimation plus rule-based logic and median/IQR baselines produced a more honest system than forcing a weak ML layer.

### Visual validation is necessary

Metrics alone are not enough. Key frames, plots, and evaluation reports are needed to inspect whether the pipeline is telling the truth.

---

## Future Work

Potential next steps:

- trim videos automatically to one swing window
- improve event detection with confidence checks
- fix shoulder angle wraparound
- save both normalized and non-normalized cleaned landmarks
- improve hip drift using non-normalized coordinates
- add pose overlays to key event frames
- separate movement, timing, and angle visualizations more cleanly
- collect a larger and cleaner professional baseline
- collect coach-labeled swing issues
- add ML only when enough labeled data exists

Potential ML extensions:

- swing phase classification
- anomaly detection on extracted feature vectors
- swing similarity clustering
- supervised flaw classification from coach labels

---

## Final Project Summary

This project evolved from a one-video MVP into a modular video-analysis pipeline that can process multiple professional swings, generate a pose-derived baseline, and compare unseen test swings against that baseline.

The strongest takeaway is that good applied AI systems are not just models. They require data quality checks, modular architecture, reproducible outputs, honest limitations, and clear evidence for every claim.
