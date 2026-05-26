# Baseball Swing Analysis Capstone Plan

**Project start date:** 2026-04-22  
**Owner:** Andrew Bieber  
**Project type:** Explainable motion analysis / computer vision / applied ML capstone  
**Status:** In planning

---

# 1. Project Summary

## One-sentence definition
Build an explainable baseball swing analysis system that takes in a recorded swing video, extracts body movement over time, measures key mechanics, and returns evidence-backed feedback.

## Project description
This project aims to build a staged baseball swing analysis pipeline, starting with a narrow and realistic MVP. The system will take a recorded swing video, extract pose landmarks over time, transform those landmarks into interpretable movement features, detect important swing phases, and generate explainable feedback supported by visual and numerical evidence. The long-term goal is not just to label a swing as “good” or “bad,” but to measure mechanics such as head movement, stride timing, front-side stability, sequencing, and hand path, then justify conclusions with plots, overlays, and clear logic.

---

# 2. Core Engineering Principle

This project will be built in stages.

**Rule:** Do not build the final dream system immediately.  
**Rule:** Build the smallest working version that proves the project is real.  
**Rule:** Prefer explainable, measurable outputs over flashy AI claims.

---

# 3. Final Vision vs MVP

## Final Vision
The full version of this project would:

- accept a swing video from a phone or camera
- detect and track the hitter over time
- identify key swing phases automatically
- measure important mechanics such as:
  - head movement
  - stride timing
  - hip/shoulder rotation
  - hand path
  - front-side stability
  - sequencing of lower half vs upper half
- generate feedback such as:
  - "your head moved too much"
  - "your front side did not stabilize before contact"
  - "your upper half fired before the lower half finished working"
- provide evidence:
  - annotated frames
  - landmark overlays
  - motion plots
  - numeric metrics
- later recommend possible corrective ideas or drills

## MVP
The first real version will do much less:

- support **one camera angle only**
- process **one prerecorded swing video**
- extract body pose landmarks
- smooth and normalize landmark trajectories
- compute **3 core mechanics metrics**
- generate **1 to 3 rule-based feedback statements**
- show supporting evidence visually and numerically

## MVP Target
The MVP should answer questions like:

- How much did the head move during the swing window?
- Did the front side stabilize before contact?
- Did lower-half movement begin before upper-body/hand acceleration?

---

# 4. Scope Constraints

## In Scope for Early Versions
- prerecorded video
- one hitter in frame
- one view only
- pose extraction using an existing model/library
- NumPy-based feature calculations
- simple rule-based phase detection
- explainable metrics and feedback
- static visualizations and reports

## Out of Scope for Early Versions
- real-time analysis
- multiple camera angles
- custom pose model training
- bat tracking
- ball tracking
- mobile app
- full biomechanics-grade validation
- autonomous drill recommendation engine
- open-ended chatbot interface

---

# 5. System Pipeline

## High-level pipeline
1. Input video
2. Video preprocessing
3. Pose landmark extraction
4. Landmark smoothing and cleaning
5. Coordinate normalization
6. Event / phase detection
7. Feature extraction
8. Rule-based feedback generation
9. Visualization / report generation

## Expanded pipeline

### A. Input
- swing video file
- optional metadata:
  - hitter handedness
  - frame rate
  - camera angle

### B. Preprocessing
- load video
- trim swing window if necessary
- sample frames
- resize or crop if needed
- reject bad input if necessary

### C. Pose Extraction
- run pose estimator
- collect landmarks per frame
- store confidence scores
- save results in structured format

### D. Landmark Processing
- smooth noisy keypoints
- handle missing values
- normalize positions relative to body size or stance
- build clean trajectories over time

### E. Event / Phase Detection
Early target phases or events:
- setup
- first move / load
- stride plant proxy
- contact proxy
- post-contact

### F. Feature Extraction
Candidate features:
- head displacement
- lead side stabilization proxy
- pelvis rotation proxy
- shoulder rotation proxy
- hand path proxy
- lower-half vs upper-half timing relationship

### G. Feedback Logic
Convert metrics into statements using rules:
- thresholds
- event ordering
- confidence checks
- evidence references

### H. Output
- metrics summary
- feedback summary
- plots
- annotated key frames

---

# 6. Module Breakdown

## Module 1: Video Ingestion
**Purpose:** Load and manage raw video input.

### Software Utilized

**OpenCV**
- free and easy to ingest videos and capture key frames
- simple and straightforward
- easy stack to google, learn, and apply
- No need for any advanced pose-estimates etc
- Just want to properly ingest a video, extract key frames

### Responsibilities
- load video file     # Completed 4/23/26 loaded with openCV via io.video_loader.py called in src.main.py
- extract frames        # Completed 4/23/26 loaded with openCV via io.video_loader.py called in src.main.py
- keep frame timestamps   # Completed 4/23/26 loaded with openCV via io.video_loader.py called in src.main.py
- basic metadata handling    # Completed 4/23/26 loaded with openCV via io.video_loader.py called in src.main.py

### Outputs
- frame list / iterator    # Completed 4/25/26 completed frame_saver.py which saves debug frames of first, middle, and end
- fps                      # Completed 4/25/26 completed frame_saver.py which saves debug frames of first, middle, and end
- frame count              # Completed 4/25/26 completed frame_saver.py which saves debug frames of first, middle, and end
- timestamps               # Completed 4/25/26 completed frame_saver.py which saves debug frames of first, middle, and end

---

### Module 1 Completion Notes — Video Ingestion

**Completed:** 4/25/26

Module 1 is complete for the first version of the project. The pipeline can now load a raw swing video with OpenCV, read basic metadata, extract frames, attach frame timestamps, and save debug frames for visual inspection.

#### What was built
- `load_video_metadata(video_path)`
  - loads a video file
  - reads FPS, frame count, resolution, and duration
- `extract_frames_with_timestamps(video_path)`
  - extracts each frame from the video
  - assigns each frame a frame index
  - computes timestamp using `frame_index / fps`
  - stores each frame as a record with index, timestamp, and frame data
- `save_frame_image(frame, output_path)`
  - saves a single frame image to disk
- `save_debug_frames(frames_with_timestamps, output_dir)`
  - saves the first, middle, and last frame for sanity-checking

#### Important observation
The debug frame output showed that the last two selected frames looked nearly identical because the source recording contained extra dead time / repeated end frames. This was useful because it revealed an important project lesson early: **data quality directly affects every downstream stage**.

If the input video contains unnecessary dead time, repeated frames, bad trimming, poor camera angle, or shaky footage, later modules such as pose estimation, swing phase detection, and feature extraction can produce misleading results.

#### Engineering lesson
The ingestion pipeline is not just about loading a video. It also needs to help verify whether the video is usable. Saving debug frames made it obvious that the input clip quality matters before any ML or motion analysis is attempted.

#### Design implication
Before moving too far into pose extraction, the project should support some simple way to control the analyzed section of the video, such as:
- selecting a start frame and end frame
- trimming the video to only the actual swing
- saving debug frames from the selected swing window

This does not need to be a full video editor yet, but the system should not blindly analyze dead frames or irrelevant footage.

#### Status
Module 1 is complete enough to move forward.


## Module 2: Pose Extraction Interface
**Purpose:** Run an external pose model and standardize output.

### Responsibilities
- call pose estimator     # Completed 4/27/26 completed pose_estimator.py which contains the classes and methods to analyze each frame to apply a pose
- extract key landmarks per frame      # Completed 4/27/26 completed pose_estimator.py which contains the classes and methods to analyze each frame to apply a pose
- collect confidence values        # Completed 4/27/26 completed pose_estimator.py which contains the classes and methods to analyze each frame to apply a pose
- save raw keypoint output         # Completed 4/27/26 completed pose_estimator.py which contains the classes and methods to analyze each frame to apply a pose

### Outputs
- per-frame landmark arrays # Completed 4/27/26 completed pose_pipeline.py which processes each frame and estimates each 33 landmarks and 4 components (x,y,z,confidence) 
- confidence arrays # Completed 4/27/26 completed pose_pipeline.py which processes each frame and estimates each 33 landmarks and 4 components (x,y,z,confidence)
- raw pose data file  # Completed 4/27/26 completed pose_pipeline.py which processes each frame and estimates each 33 landmarks and 4 components (x,y,z,confidence)

### Module 2 Completion Notes — Pose Extraction Interface

**Completed:** 4/27/26

Module 2 is complete for the first version of the project. The system can now take the frames from a raw baseball swing video, run a pose estimation model on each frame, draw visual debug landmarks, and save the raw landmark data into a structured NumPy `.npz` file.

The completed output for the current test video produced:

```txt
66 frames × 33 landmarks × 4 components

Saved landmark shape:

(66, 33, 4)

Meaning:

axis 0 = frame over time
axis 1 = MediaPipe landmark index
axis 2 = x, y, z, visibility
What was built
pose_estimator.py
contains PoseFrameResult
contains MediaPipePoseEstimator
runs MediaPipe Pose on one frame
returns one frame’s landmark data as a structured result
can draw pose landmarks on a debug frame
pose_pipeline.py
opens the video
loops through every frame
calls the pose estimator
collects landmark arrays and confidence/visibility values
saves debug images
stacks all frames into one (num_frames, 33, 4) array
saves raw pose data to .npz
main.py
launches the Module 2 pose extraction pipeline with the selected video paths
Architecture lesson

This module showed why separating files matters.

pose_estimator.py = process one frame
pose_pipeline.py = process the full video
main.py = launch the pipeline

This made the project easier to reason about because each file has one job.

Key idea:

Class = reusable tool/object
Method = action attached to that object
Pipeline = sequence of steps using those tools
Main = launch point
Python class lesson

The MediaPipePoseEstimator class stores the MediaPipe model internally.

Instead of the pipeline needing to understand every MediaPipe detail, it can simply call:

result = estimator.estimate_frame(frame, frame_index)

The PoseFrameResult dataclass acts like a clean container for one frame’s pose result:

frame_index
landmarks: shape (33, 4)
pose_detected

This helped clarify how classes and dataclasses can organize code and data in a larger project.

Why I Chose MediaPipe

MediaPipe was chosen because this is a single-person baseball swing analysis project.

The 33 landmarks give more detailed body information than smaller pose models, which should help later with:

head movement
shoulder movement
hip movement
wrist/hand path
stride behavior
front-side stability
posture through contact and follow-through

The debug images confirmed MediaPipe tracked the hitter well enough to continue.

Stack compatibility lesson

A major lesson was that software stacks are version-dependent.

The first setup used:

Python 3.13
MediaPipe 0.10.35

That failed because the needed mp.solutions.pose API was not available.

The working setup became:

Python 3.10.11
MediaPipe 0.10.14
OpenCV
NumPy
project-specific .venv

Important dependency pin:

mediapipe==0.10.14

Main lesson:

A stack is not just a library.
It also includes Python version, package version, OS, virtual environment, and API compatibility.
Current limitations
hand/wrist landmarks may be noisy during fast motion
the bat is not tracked
occlusion can affect hands and arms
landmarks may jitter between frames
video quality and camera angle affect accuracy
Next step

Module 3 should focus on processing the raw landmark data before making swing claims.

Planned Module 3 focus:

load .npz pose data
name important landmarks
separate coordinates from visibility/confidence
inspect unreliable points
smooth noisy landmark paths if needed
prepare clean data for feature extraction
Status

Module 2 is complete enough to move forward.

raw swing video
→ frame-by-frame pose estimation
→ debug skeleton images
→ saved NumPy landmark data

---

### Module 3 Completion Notes**Completed:** 
4/30/26

Module 3 is complete for the first version of the project. The system can now load the raw pose output from Module 2, extract the landmark array, smooth landmark movement, normalize the coordinates around the hitter’s body, and save a cleaned landmark file.

#### What was built
- `load_landmarks(input_path)`
- loads the saved Module 2 `.npz` pose file  
- extracts the `landmarks` array from the archive
- `inspect_landmarks(landmarks)`  
- prints landmark shape, frame count, landmark count, and value range
- `smooth_trajectory(trajectory, window_size)`  
- smooths one landmark trajectory over time using a moving average
- `smooth_all_landmarks(landmarks, window_size)`  
- applies smoothing to all 33 landmarks across all frames
- `get_body_center(landmarks)`  
- calculates hip center using left hip and right hip landmarks
- `normalize_to_body_center(landmarks)`  
- subtracts hip center from all landmark positions
- `process_landmarks(input_path, output_path, smoothing_window)`  
- runs the full Module 3 pipeline#### Engineering decisions
- Module 2 saved pose data as a `.npz` file, not a direct `.npy` array.  
- The `.npz` contains `landmarks`, `frame_indices`, and `pose_detected`.  
- Module 3 specifically loads `landmarks`.
- The landmark shape is:```text(num_frames, num_landmarks, 4)

For the test video:
(66, 33, 4)

Meaning:

66 frames33 MediaPipe landmarks4 values: x, y, z, visibility

Smoothing uses a moving average over nearby frames.

Too small of a window keeps jitter.

Too large of a window can blur fast swing movement.

Current smoothing choice:

smooth_trajectory() default = 5

smooth_all_landmarks() default = 3

process_landmarks() uses 3


I chose 3 for the full pipeline because a baseball swing is fast, and preserving timing matters more than making the data overly smooth.

Visibility is not smoothed.

x, y, z are position values.

visibility is a confidence value.


Normalization uses hip center:

hip_center = (left_hip + right_hip) / 2
This makes movement body-relative instead of screen-relative.
Output
data/processed/pose/mike_trout_swing_01_pose_cleaned.npy
Status
Module 3 is complete enough to move forward to Module 4: Event / Phase Detection.

---

## Module 4: Event / Phase Detection
**Purpose:** Identify useful time points in the swing.

### Responsibilities
- detect movement start
- detect stride plant proxy
- detect contact proxy
- label rough phases

### Outputs
- event frame indices
- phase labels / time windows

### Module 4 Completion Notes

**Completed:** 5/1/26

Module 4 is complete for the first version of the project. The system can now load cleaned landmark data from Module 3, calculate a wrist-based hand speed proxy, detect rough swing event frames, and save those events to a JSON file.

#### What was built

- `event_detector.py`
  - defines `SwingEvents`
  - defines `SwingEventDetector`
  - extracts wrist landmarks
  - calculates frame-to-frame wrist speed
  - detects movement start
  - detects peak hand speed
  - estimates contact proxy

- `event_pipeline.py`
  - loads cleaned landmark data
  - runs the event detector
  - saves detected events as JSON

#### Engineering decisions

- Wrist movement is used as the first v1 swing signal.
  - MediaPipe does not track the bat or ball.
  - The wrists are the closest available body landmarks to the bat/hands.

- Hand speed proxy uses the max speed between left and right wrists.
  - This avoids needing to know hitter handedness in v1.

- Movement start is detected when hand speed passes a percentage of max hand speed.

- Peak hand speed is the frame with the highest wrist-speed proxy.

- Contact is currently a proxy, not true contact detection.
  - Since bat/ball contact is not tracked, contact is estimated a few frames after peak hand speed.

#### Test Output

```json
{
    "movement_start": 3,
    "peak_hand_speed": 18,
    "contact_proxy": 21
}
Status

Module 4 is complete enough to move forward to Module 5: Feature Extraction.

---

## Module 5: Feature Extraction

**Purpose:** Compute interpretable baseball swing mechanics metrics from cleaned pose landmarks and detected swing events.

Module 5 is the first module where the project starts turning raw movement data into actual swing measurements. The goal is not to give coaching advice yet. The goal is to calculate useful numbers that can later be interpreted by a feedback module.

### Responsibilities

- calculate motion-based swing features
- compare body regions over time
- derive timing relationships between detected swing events
- produce metrics with clear units or normalized units
- save the extracted features in a readable format

### Outputs

- feature dictionary
- saved feature JSON file
- frame-based timing values
- normalized movement values
- angle-based rotation proxy values

---

### Module 5 Completion Notes

**Completed:** 5/18/26

Module 5 is complete for the first version of the project. The system can now load cleaned pose landmarks from Module 3, load detected swing events from Module 4, calculate basic swing mechanics features, and save the results as a JSON file.

This module helped connect the earlier parts of the pipeline:

```text
Module 3 cleaned landmarks
+ Module 4 detected events
→ Module 5 extracted swing features
```

---

### What was built

- `feature_extractor.py`
  - defines `SwingFeatures`
  - defines `SwingFeatureExtractor`
  - stores cleaned landmark data and detected event frames
  - calculates head movement from movement start to contact proxy
  - calculates average hand path distance from movement start to contact proxy
  - calculates hip drift from movement start to contact proxy
  - calculates shoulder angle change from movement start to contact proxy
  - calculates timing relationships between swing events

- `feature_pipeline.py`
  - loads cleaned landmark data
  - loads detected event JSON data
  - creates the `SwingFeatureExtractor` object
  - runs all feature calculations
  - saves extracted features as a JSON file

- `main.py`
  - temporarily runs the Module 5 feature extraction pipeline
  - prints the extracted features to the terminal for inspection

---

### Current input files

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

The event JSON contains detected swing event frames:

```json
{
    "movement_start": 3,
    "peak_hand_speed": 18,
    "contact_proxy": 21
}
```

---

### Current output file

```text
data/processed/features/mike_trout_swing_01_features.json
```

Example output from the first test run:

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

### Engineering decisions

The main design decision in this module was modularity.

The feature extraction logic was separated from the pipeline logic:

```text
feature_extractor.py = owns the swing feature calculations
feature_pipeline.py = owns loading files, running the extractor, and saving output
main.py = launch point for testing the module
```

This kept the project easy to reason about.

The `SwingFeatureExtractor` class acts as the main object for Module 5. It stores:

```text
cleaned landmark array
detected swing event frames
feature calculation methods
```

This made the module feel organized because each method calculates one specific swing metric.

---

### Why NumPy fits this module

NumPy works well here because the swing data is already structured as arrays.

The main landmark array is:

```text
frames × landmarks × values
```

For the current test video:

```text
66 frames × 33 landmarks × 4 values
```

This means the full swing can be treated as motion data over time.

Even when a feature returns one number, that number comes from calculations across multiple frames.

Examples:

```text
head movement = distance between nose position at movement start and contact proxy
hand path = total wrist movement across consecutive frames
shoulder angle change = angle difference between two event frames
timing = frame difference between detected events
```

This showed why NumPy is useful even when the final output is a single number.

---

### Current feature meanings

#### Head Movement

```text
head_movement_start_to_contact
```

Measures how far the nose landmark moved from movement start to contact proxy.

This is a v1 proxy for head stability.

The first test produced a very small value, which directionally makes sense because the sample swing is Mike Trout and his head movement should be controlled.

---

#### Hand Path Distance

```text
hand_path_start_to_contact
```

Measures the average wrist path distance from movement start to contact proxy.

This uses both wrist landmarks and averages their total movement path.

This should normally be much larger than head movement because the hands are supposed to move during the swing while the head should remain more stable.

---

#### Hip Drift

```text
hip_drift_start_to_contact
```

Measures hip center movement from movement start to contact proxy.

The first test produced a value near zero:

```text
1.4901161193847656e-08
```

This revealed an important issue.

Module 3 normalized every frame around the hip center. Because of that, the hip center is basically forced to stay near the origin. This means hip drift cannot be measured correctly from the normalized landmark file.

This is not a math bug. It is a data-design issue.

Design implication:

```text
normalized landmarks = useful for body-relative mechanics
non-normalized cleaned landmarks = needed for whole-body movement features
```

Future improvement:

```text
Save both cleaned non-normalized landmarks and normalized landmarks.
```

Then use:

```text
normalized data → body-relative features
non-normalized cleaned data → body translation / drift features
```

---

#### Shoulder Angle Change

```text
shoulder_angle_change_start_to_contact
```

Measures the angle change of the shoulder line from movement start to contact proxy.

This uses the line between the left shoulder and right shoulder landmarks.

Unlike the distance metrics, this output is in degrees.

The first test produced:

```text
48.16871643066406 degrees
```

This is directionally believable for a shoulder rotation proxy during a swing.

---

#### Timing Features

```text
frames_start_to_contact
frames_start_to_peak_hand_speed
frames_peak_hand_speed_to_contact
```

These values measure timing relationships between detected swing events.

The first test produced:

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

These frame-based timing values are useful because they connect the swing events into measurable relationships.

---

### Unit clarification

The distance-based metrics are not inches, centimeters, feet, or meters.

They are currently measured in normalized landmark coordinate units.

This means the values are best used for relative comparison, such as:

```text
head movement compared to hand path
movement during one phase compared to another phase
body part stability within the same video
comparison between videos processed the same way
```

They should not yet be treated as exact real-world physical measurements.

Angle values are in degrees.

Timing values are in frames.

---

### Important observation from testing

The first Module 5 test showed that several metrics are already directionally useful:

```text
head movement
hand path distance
shoulder angle change
timing relationships
```

Since the sample video is Mike Trout, the low head movement and clean timing seemed believable.

The biggest discovery was that hip drift is not useful yet because the landmark data had already been normalized around the hip center.

This helped clarify that different features may require different versions of the landmark data.

---

### Project ownership lesson

Module 5 showed that the project is becoming easier to understand because of the modular structure.

At this point, the project is not just copied code. The design is becoming clear:

```text
Module 1 = get the video
Module 2 = extract pose landmarks
Module 3 = clean and normalize the landmarks
Module 4 = detect important swing events
Module 5 = measure what happened during those events
```

This module also showed the value of daily coding practice. The Python is becoming easier to read, the class structure makes more sense, and the project feels more owned instead of just generated.

The main lesson from Module 5 is:

```text
Good software structure makes the math easier to test.
Good testing makes the design decisions clearer.
```

---

### Status

Module 5 is complete enough for v1.

Current status:

```text
cleaned pose landmarks
+ detected swing events
→ extracted swing feature JSON
```

The next step is to visually compare the metrics against the swing video and decide which features are useful enough to keep.

Module 5 should not be overbuilt yet. The current version is enough to support the next stage: feedback and interpretation.

---

## Module 6: Feedback Engine

**Purpose:** Turn extracted swing metrics into evidence-backed feedback.

Module 6 takes the feature values from Module 5 and applies simple rule-based logic to generate readable feedback. The goal of this module is not to be perfect or scientifically final yet. The goal is to prove that the system can take swing measurements and turn them into useful explanations with numeric evidence.

### Responsibilities

- apply rules to extracted features
- generate feedback statements
- attach numeric evidence to each statement
- attach confidence levels
- attach warning flags for known limitations
- keep feedback separate from feature calculation

### Outputs

- feedback statements
- explanation text
- issue / warning / good status flags
- numeric evidence
- confidence labels
- known limitation warnings

---

### Module 6 Completion Notes

**Completed:** 5/18/26

Module 6 is complete for the first version of the project. The system can now take the extracted feature dictionary from Module 5 and generate rule-based feedback using hardcoded v1 thresholds.

This module connects:

```text
Module 5 extracted swing features
→ Module 6 feedback statements
```

Module 5 answers:

```text
What are the numbers?
```

Module 6 answers:

```text
What do the numbers suggest?
```

---

### What was built

- `feedback_engine.py`
  - defines `FeedbackEngine`
  - receives the feature dictionary from Module 5
  - evaluates head stability
  - evaluates hand path distance
  - evaluates shoulder rotation proxy
  - evaluates swing timing
  - adds known limitation warnings
  - generates a summary
  - returns feedback as a dictionary

- `main.py`
  - temporarily runs Module 5 feature extraction
  - passes the feature output into Module 6
  - prints summary, feedback statements, evidence, confidence, and warnings

---

### Current input

Module 6 currently receives the feature dictionary created by Module 5.

Example input:

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

### Current output

Module 6 produces a feedback dictionary with this general structure:

```text
summary
feedback
warnings
```

Each feedback item includes:

```text
metric
status
statement
evidence
confidence
```

Example structure:

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

### Current feedback checks

#### Head Stability

Metric:

```text
head_movement_start_to_contact
```

Purpose:

```text
Evaluate whether the hitter's head stayed relatively stable from movement start to contact proxy.
```

Current v1 thresholds:

```text
good: value < 0.05
warning: 0.05 <= value <= 0.10
issue: value > 0.10
```

The current Mike Trout sample was well within the good range, which makes sense visually because elite hitters usually keep the head controlled through the swing.

---

#### Hand Path

Metric:

```text
hand_path_start_to_contact
```

Purpose:

```text
Evaluate whether the hands took a reasonable movement path from movement start to contact proxy.
```

Current v1 thresholds:

```text
good: value < 0.45
warning: 0.45 <= value <= 0.65
issue: value > 0.65
```

This is a v1 hand path efficiency proxy. It is not a perfect swing-quality metric yet, but it gives the system a way to flag swings where the hands appear to travel a longer path.

---

#### Shoulder Rotation Proxy

Metric:

```text
shoulder_angle_change_start_to_contact
```

Purpose:

```text
Evaluate whether shoulder angle change is within an expected range from movement start to contact proxy.
```

Current v1 thresholds:

```text
good: 25 <= value <= 65 degrees
warning: 15 <= value < 25 degrees
warning: 65 < value <= 80 degrees
issue: value < 15 degrees or value > 80 degrees
```

This metric uses the line between the left and right shoulder landmarks as a simple v1 rotation proxy.

---

#### Timing

Metric:

```text
frames_start_to_contact
```

Purpose:

```text
Evaluate whether the time from movement start to contact proxy is within a believable frame range.
```

Current v1 thresholds:

```text
good: 10 <= value <= 25 frames
warning: 8 <= value < 10 frames
warning: 25 < value <= 35 frames
issue: value < 8 frames or value > 35 frames
```

This keeps the timing feedback simple and frame-based for now.

---

### Why hardcoded thresholds were used

Hardcoded thresholds were used because this is a v1 feedback engine.

The goal right now is not to discover perfect swing thresholds. The goal is to create a working interpretation layer that can turn feature values into readable feedback.

The current thresholds are based on:

```text
early testing
baseball reasoning
the Mike Trout sample output
directional expectations
```

These thresholds are temporary.

They should later be refined by running more swing videos through the same pipeline and comparing feature distributions.

---

### Known limitations

#### Hip drift is not evaluated yet

Metric:

```text
hip_drift_start_to_contact
```

Hip drift is not currently used for feedback.

Reason:

```text
Module 3 normalized each frame around the hip center.
```

Because of that, the hip center is basically forced to stay near the origin. This makes hip drift appear almost zero by design.

This is not a feedback engine issue. It is a data representation issue.

Future fix:

```text
Save both normalized landmarks and non-normalized cleaned landmarks.
```

Then use:

```text
normalized landmarks → body-relative mechanics
non-normalized cleaned landmarks → body translation / drift metrics
```

---

### Engineering decisions

The main design decision was to keep Module 6 simple.

A separate pipeline file was not needed yet because Module 6 only interprets the feature dictionary produced by Module 5.

The current structure is:

```text
feedback_engine.py = owns feedback logic
main.py = temporary test launcher
```

This is enough for v1 because the module is only doing rule-based interpretation.

The long file length mainly comes from repeated if-else logic and detailed feedback statements, not from complicated architecture.

---

### Why this module matters

Module 6 is where the project starts to become useful to a person.

Before this module, the system could calculate numbers.

After this module, the system can explain those numbers.

The project flow is now:

```text
video
→ pose landmarks
→ cleaned landmarks
→ detected events
→ extracted features
→ evidence-backed feedback
```

This is a major step because the system is no longer just processing data. It is starting to produce human-readable analysis.

---

### Future improvement path

The long-term plan for feedback is:

```text
V1: hardcoded if-else thresholds
V2: thresholds refined using more swing samples
V3: dynamic thresholds based on feature distributions
V4: ML-assisted swing evaluation
```

The side-view college swing clips can eventually be used as a test set or validation set.

Those swings can be processed through the same pipeline to build a feature table:

```text
player
head_movement
hand_path
shoulder_angle_change
frames_start_to_contact
feedback status
```

That data can later help improve the thresholds or train a model to recognize stronger and weaker swing patterns.

---

### Project ownership lesson

Module 6 was quick to build because the structure was already clear.

The important part was understanding what the module should own:

```text
Module 5 calculates metrics.
Module 6 interprets metrics.
```

This showed why modularity matters. Since the previous modules were separated cleanly, adding the feedback engine was straightforward.

The if-else logic is simple, but it fits the current stage of the project. The complexity can come later after the full v1 pipeline works end to end.

The main lesson from Module 6 is:

```text
Simple logic is fine when the module responsibility is clear.
```

---

### Status

Module 6 is complete enough for v1.

Current status:

```text
Module 5 feature dictionary
→ Module 6 rule-based feedback
→ summary, feedback statements, evidence, and warnings
```

The next step is Module 7: Visualization / Debug Output.

Module 7 should visually prove the numbers and feedback by drawing useful information over the swing frames or output images.

---

## Module 7: Visualization / Report Assets

**Purpose:** Show visual proof for the detected events, extracted features, and generated feedback.

Module 7 creates visual outputs from the data already produced by earlier modules. The goal is not to add new swing logic. The goal is to make the system easier to inspect, debug, and eventually present in a final report/demo.

### Responsibilities

- create key-frame images for detected swing events
- create trajectory plots for important landmarks
- create metric / feature summary charts
- create timing event visualizations
- save visual assets for debugging and reporting
- provide visual proof for earlier module outputs

### Outputs

- key-frame images
- trajectory plots
- timing event chart
- feature summary chart
- report/debug assets

---

### Module 7 Completion Notes

**Completed:** 5/25/26

Module 7 is complete for the first version of the project. The system can now load the original video, cleaned pose landmarks, detected swing events, and extracted feature values, then create visual proof assets.

This module connects:

```text
original video
+ cleaned landmarks
+ detected events
+ extracted features
→ visualization outputs
```

Earlier modules answer:

```text
Module 4 = What swing events were detected?
Module 5 = What features were calculated?
Module 6 = What feedback was generated?
```

Module 7 answers:

```text
Can we visually inspect and prove those results?
```

---

### What was built

- `visualizer.py`
  - defines `SwingVisualizer`
  - saves key event frames from the original video
  - plots hand path trajectory
  - plots head movement trajectory
  - plots a feature summary chart
  - plots a detected event timeline
  - saves all visualization outputs to a selected output folder

- `visualization_pipeline.py`
  - loads cleaned landmark data
  - loads detected event JSON data
  - loads extracted feature JSON data
  - creates the `SwingVisualizer` object
  - runs all visualization methods
  - saves visual outputs to disk

- `main.py`
  - temporarily runs Module 5 feature extraction
  - runs Module 6 feedback generation
  - runs Module 7 visualization generation
  - prints the visualization output folder

---

### Current input files

```text
data/raw/mike_trout_swing_01.mp4
data/processed/pose/mike_trout_swing_01_pose_cleaned.npy
data/processed/events/mike_trout_swing_01_events.json
data/processed/features/mike_trout_swing_01_features.json
```

---

### Current output folder

```text
outputs/visualizations/mike_trout_swing_01/
```

Current output files:

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

### Current visualizations

#### Key Event Frames

Files:

```text
key_frame_movement_start.jpg
key_frame_peak_hand_speed.jpg
key_frame_contact_proxy.jpg
```

Purpose:

```text
Show the actual video frames where the event detector marked important swing moments.
```

Current detected events:

```text
movement_start = frame 3
peak_hand_speed = frame 18
contact_proxy = frame 21
```

These images were useful because they allowed visual inspection of whether Module 4 detected reasonable event frames.

For the Mike Trout sample, the key frames looked directionally believable.

---

#### Timing Event Timeline

File:

```text
timing_events.png
```

Purpose:

```text
Show detected swing events on a simple frame-index timeline.
```

This chart helps visualize the spacing between:

```text
movement start
peak hand speed
contact proxy
```

For the current sample:

```text
movement_start → frame 3
peak_hand_speed → frame 18
contact_proxy → frame 21
```

This supports the timing metrics calculated in Module 5.

---

#### Head Movement Trajectory

File:

```text
head_path_plot.png
```

Purpose:

```text
Plot the nose/head landmark path across frames.
```

This visual supports:

```text
head_movement_start_to_contact
```

The plot showed some expected landmark noise, but the measured value was still small for the movement start to contact proxy window.

Important note:

```text
MediaPipe landmarks can jitter slightly frame-to-frame.
```

This is especially true during fast motion, blur, or partial occlusion.

---

#### Hand Path Trajectory

File:

```text
hand_path_plot.png
```

Purpose:

```text
Plot left and right wrist movement paths across frames.
```

This visual supports:

```text
hand_path_start_to_contact
```

The hand path plot showed more motion than the head path, which makes sense because the hands travel aggressively during the swing while the head should remain more controlled.

Some noise is expected because wrists move quickly and are more likely to blur or be partially occluded.

---

#### Feature Summary Chart

File:

```text
feature_summary.png
```

Purpose:

```text
Show selected Module 5 feature values in one quick debug chart.
```

Current plotted values:

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

Because of this, the `Value` axis is only a raw numeric value axis.

The bars should not be compared directly as if they are measuring the same kind of quantity.

For example, shoulder angle dominates the chart because a value like `48 degrees` is numerically much larger than a normalized movement value like `0.35`.

This chart is useful as a quick debug view, but it is not a clean scientific comparison chart yet.

Future improvement:

```text
distance_metrics.png
angle_metrics.png
timing_metrics.png
```

This would separate charts by unit type.

---

### Engineering decisions

The main design decision was to keep visualization separate from calculation and feedback.

The structure is:

```text
Module 5 = calculate features
Module 6 = interpret features
Module 7 = visualize proof
```

This keeps each module responsible for one major job.

Module 7 does not change feature values or feedback logic. It only displays data that previous modules already produced.

---

### Why two files were enough

Two main files were enough for this module:

```text
visualizer.py = owns plotting and frame-saving methods
visualization_pipeline.py = owns loading data and running the visualizer
```

This matches the existing project pattern:

```text
tool/class file = reusable logic
pipeline file = connects inputs and outputs
```

This module did not need more structure because the graphs are simple and the purpose is visual debugging.

---

### Why Python was enough

Python is enough for v1 visualization.

Current tools:

```text
OpenCV = read video frames and write labeled key-frame images
Matplotlib = create plots and charts
NumPy = access landmark arrays
JSON = load event and feature data
```

No R or separate reporting tool is needed for the current version.

---

### Important observations from testing

The first Module 7 test successfully generated all expected visual assets.

The key event frames were the most immediately useful because they showed the exact video frames tied to the detected swing events.

The trajectory plots worked, but showed expected noise due to landmark jitter and fast swing movement.

The feature summary chart worked, but revealed that mixed-unit charts should be treated as debug-only.

Main observations:

```text
key event frames are useful for validating event detection
trajectory plots can show noise from landmark jitter
feature summary charts should not mix units long-term
plots should eventually focus on the movement_start → contact_proxy window
```

---

### Known limitations

#### Trajectory plots include extra frames

The current trajectory plots can include more than just the main swing window.

Future improvement:

```text
plot only movement_start → contact_proxy
```

This would make the visuals more directly tied to the calculated feature values.

---

#### Feature summary chart mixes units

The current feature summary chart mixes:

```text
normalized movement units
degrees
frames
```

This makes the chart useful for quick debugging but not for direct metric comparison.

Future improvement:

```text
separate charts by metric type / unit type
```

---

#### Key frames do not yet include pose overlays

The current key event images only show the raw frame with a text label.

Future improvement:

```text
draw pose landmarks or tracked points on key event frames
```

This would make the visual proof stronger.

---

### Future improvements

Possible improvements after v1:

```text
plot only the event window instead of all frames
separate distance, angle, and timing charts
draw pose landmarks on key frames
draw wrist/head trajectory directly on video frames
create side-by-side event frame comparison image
add feedback text onto visualization outputs
create an annotated video clip
generate a simple HTML or Markdown report
```

The most useful near-term improvements are:

```text
1. Restrict trajectory plots to movement_start → contact_proxy.
2. Split mixed-unit charts by unit type.
3. Add landmark overlays to key event frames.
```

---

### Project ownership lesson

Module 7 showed why visual proof matters.

The earlier modules produced numbers and feedback, but the visualizations made those outputs easier to inspect.

The module also revealed limitations that were not obvious from the numbers alone.

This is the point of visualization:

```text
not just to make the project look better,
but to help validate whether the pipeline is telling the truth.
```

The main lesson from Module 7 is:

```text
Visual debugging turns abstract metrics into inspectable evidence.
```

---

### Status

Module 7 is complete enough for v1.

Current status:

```text
previous module outputs
→ visualization pipeline
→ saved proof assets
```

The next step is Module 8: final end-to-end demo / report packaging.

Module 8 should focus on connecting the full v1 pipeline into one clean final run and summarizing the project output.

---

## Module 8: Evaluation

**Purpose:** Verify whether the v1 pipeline results are stable, believable, and honestly documented.

Module 8 evaluates the outputs from the completed v1 pipeline. The goal is not to add new swing mechanics or new feedback logic. The goal is to check whether the system ran end-to-end, whether the outputs are believable, and whether the current limitations are clearly documented.

### Responsibilities

- inspect whether the full pipeline runs end-to-end
- review detected event frames visually
- inspect whether feature values are directionally believable
- verify that feedback statements match the extracted metrics
- document assumptions, limitations, and failure cases
- prepare the system for future multi-video evaluation

### Outputs

- evaluation notes
- single-sample evaluation report
- failure / limitation log
- future validation plan

---

### Module 8 Completion Notes

**Completed:** 5/25/26

Module 8 is complete for the first version of the project. The system can now take outputs from the previous modules and create a simple markdown evaluation report.

This module connects:

```text
detected events
+ extracted features
+ feedback output
+ visualization assets
→ evaluation report
```

The evaluation report is currently saved as:

```text
outputs/report/mike_trout_swing_01_evaluation.md
```

---

### What was built

- `evaluator.py`
  - defines `SwingEvaluator`
  - checks whether detected events occur in a believable order
  - checks whether required feature values exist
  - checks whether feature values are directionally believable
  - checks whether feedback output has the expected structure
  - checks whether visualization files were created
  - documents known limitations
  - builds a markdown evaluation report
  - saves the report to disk

- `main.py`
  - now runs Module 5 feature extraction
  - runs Module 6 feedback generation
  - runs Module 7 visualization output
  - runs Module 8 evaluation report generation

---

### Current input sources

Module 8 uses outputs from earlier modules:

```text
data/processed/events/mike_trout_swing_01_events.json
data/processed/features/mike_trout_swing_01_features.json
outputs/visualizations/mike_trout_swing_01/
```

It also receives the Module 6 feedback output during the current run.

---

### Current output

```text
outputs/report/mike_trout_swing_01_evaluation.md
```

The report includes:

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

### Evaluation scope

This is a single-sample evaluation.

The current v1 system only uses one active test video:

```text
mike_trout_swing_01.mp4
```

Because of that, Module 8 does not claim full accuracy or full validation across many hitters.

The purpose is to confirm:

```text
Does the pipeline work end-to-end on one sample?
Are the outputs believable?
Are the limitations documented honestly?
```

This is better described as:

```text
single-sample pipeline evaluation
```

not:

```text
full dataset validation
```

---

### Current evaluation checks

#### Event Order Check

Checks whether the detected events occur in this order:

```text
movement start → peak hand speed → contact proxy
```

For the current Mike Trout sample:

```text
movement_start = frame 3
peak_hand_speed = frame 18
contact_proxy = frame 21
```

This order is believable for the v1 event detection system.

---

#### Feature Value Check

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
shoulder angle change should be within a broad believable range
movement start to contact should be within a broad believable frame range
```

This does not prove the metrics are perfect. It only confirms that the current output is not obviously broken.

---

#### Feedback Output Check

Checks whether Module 6 produced the expected feedback structure:

```text
summary
feedback
warnings
```

This confirms that the feedback engine created readable statements with evidence and warning flags.

---

#### Visualization Output Check

Checks whether Module 7 created the expected visual files:

```text
key_frame_movement_start.jpg
key_frame_peak_hand_speed.jpg
key_frame_contact_proxy.jpg
hand_path_plot.png
head_path_plot.png
feature_summary.png
timing_events.png
```

This confirms that the project created visual proof assets for the v1 output.

---

### Known limitations documented

Module 8 documents the main v1 limitations:

```text
only one sample video is currently evaluated
contact is still a proxy because the bat and ball are not tracked
distance metrics are normalized coordinate units, not real-world units
hip drift is unreliable because landmarks were normalized around the hip center
feedback thresholds are hardcoded v1 placeholders
MediaPipe landmarks can jitter during fast motion, blur, or occlusion
```

These limitations are important because the v1 system should not be oversold.

---

### Why Module 8 matters

Module 8 is about honesty and validation.

The earlier modules made the system work.

Module 8 asks:

```text
What worked?
What looked believable?
What needs review?
What assumptions were made?
What should be tested next?
```

This makes the project stronger because it separates working code from proven accuracy.

The project now has an evaluation layer instead of just a pipeline that prints outputs.

---

### Engineering decisions

The main design decision was to make Module 8 a report generator instead of another mechanics module.

The structure is:

```text
Module 5 = calculate features
Module 6 = interpret features
Module 7 = visualize proof
Module 8 = evaluate and document the v1 result
```

This kept the final module focused.

Module 8 does not change feature values, feedback rules, or visual outputs. It only checks and summarizes them.

---

### Current v1 pipeline

With Module 8 complete, the full v1 pipeline is:

```text
video
→ frame ingestion
→ pose extraction
→ landmark processing
→ event detection
→ feature extraction
→ feedback engine
→ visualization assets
→ evaluation report
```

This means v1 is now complete.

---

### Future validation plan

After v1, the next step is to test the same pipeline on more side-view swing clips.

Future evaluation steps:

```text
run the full pipeline on more swing samples
save feature JSON for each swing
compare feature distributions
track when pose detection fails
track when event detection is visually wrong
track when feedback does not match the video
adjust thresholds using real swing data
build a small validation set
eventually train ML models on extracted feature vectors
```

The side-view college swing clips can become the first post-v1 validation set.

---

### Project ownership lesson

Module 8 made the project feel complete because it forced the system to explain itself.

The main lesson is:

```text
A project is not finished just because the code runs.
A project is stronger when the outputs are evaluated, limitations are documented, and the next testing path is clear.
```

This was the first full v1 pipeline that felt owned from end to end.

The project now has structure, outputs, evidence, and a future path.

---

### Status

Module 8 is complete enough for v1.

Current status:

```text
previous module outputs
→ evaluator
→ markdown evaluation report
```

V1 status:

```text
complete
```

The next phase is post-v1 improvement:

```text
clean up weak metrics
test more swing clips
refine hardcoded thresholds
improve visualizations
prepare for data-informed or ML-assisted evaluation
```
---

# 7. Rule-Based vs ML-Based Split

## Rule-Based First
Use rules for:
- event heuristics
- phase segmentation
- early metric thresholds
- feedback generation
- quality control

## ML Later
Use ML for:
- learned phase detection
- movement pattern classification
- anomaly detection
- swing similarity comparison
- feedback scoring
- feature-based issue classification

## Decision
The first working version should be primarily **rule-based + pose-based + NumPy feature engineering**.  
ML should be added only after the baseline pipeline works.

---

# 8. Architecture Decisions

## Decision 1: One camera angle only
**Choice:** Start with one view only, likely side view.  
**Reason:** Different views create different geometry and dramatically increase complexity.

## Decision 2: Use an existing pose model
**Choice:** Do not train a pose estimator.  
**Reason:** Pose is a dependency, not the capstone itself.

## Decision 3: Interpretability first
**Choice:** Build measured metrics before model-based scoring.  
**Reason:** Stronger for learning, debugging, and explaining.

## Decision 4: Narrow mechanics first
**Choice:** Start with 3 metrics, not 10+.  
**Reason:** Better chance of finishing and defending the project.

## Decision 5: Evidence must accompany feedback
**Choice:** No feedback without numeric or visual support.  
**Reason:** Prevents fake or shallow output.

---

# 9. Proposed MVP Metrics

## Metric 1: Head Movement
**Question:** How much does the head drift during the swing window?  
**Why it matters:** Excessive movement often affects vision, timing, and stability.  
**Possible representation:** Horizontal and vertical displacement normalized by torso length.

## Metric 2: Front-Side Stabilization Proxy
**Question:** Does the lead side stabilize before contact?  
**Why it matters:** Front-side stability is important for rotational transfer and consistency.  
**Possible representation:** Lead knee / lead hip / lead shoulder positional velocity near contact.

## Metric 3: Sequencing Proxy
**Question:** Does lower-half movement lead upper-half / hands?  
**Why it matters:** Good sequencing is a core swing mechanic.  
**Possible representation:** Compare timing of pelvis-related movement onset vs shoulder / hand acceleration onset.

---

# 10. Development Phases

## Phase 0 — Project Setup
**Goal:** Prepare repo, structure, and plan.

### Tasks
- [ ] Create repository structure
- [ ] Add `PLAN.md`
- [ ] Add `README.md`
- [ ] Define project scope in writing
- [ ] Define MVP in writing
- [ ] Decide initial camera angle
- [ ] Decide initial landmark set
- [ ] Create progress log section

### Done when
- repo is organized
- scope is frozen for MVP
- milestone 1 is clearly defined

---

## Phase 1 — Raw Video to Pose Data
**Goal:** Get reliable landmark output from one swing video.

### Tasks
- [ ] Select first sample swing video
- [ ] Load video frame-by-frame
- [ ] Run pose estimator on each frame
- [ ] Save landmarks per frame
- [ ] Save confidence values
- [ ] Visualize landmarks on sample frames
- [ ] Inspect whether output is usable

### Done when
- one swing video successfully produces saved body keypoints over time

---

## Phase 2 — Landmark Cleaning and Normalization
**Goal:** Convert raw keypoints into stable trajectories.

### Tasks
- [ ] Design landmark data structure
- [ ] Implement smoothing
- [ ] Handle missing values / low-confidence frames
- [ ] Choose normalization reference
- [ ] Normalize trajectories
- [ ] Plot raw vs smoothed trajectories
- [ ] Document assumptions

### Done when
- keypoint trajectories are stable enough to analyze

---

## Phase 3 — Event Detection
**Goal:** Find important swing moments.

### Tasks
- [ ] Define movement start heuristic
- [ ] Define stride plant proxy
- [ ] Define contact proxy
- [ ] Test event detection on first video
- [ ] Adjust logic after visual inspection
- [ ] Save event indices for downstream use

### Done when
- the pipeline can mark key events on at least one example swing

---

## Phase 4 — First Feature Extraction
**Goal:** Compute the first real baseball-relevant metrics.

### Tasks
- [ ] Implement head displacement metric
- [ ] Implement front-side stabilization proxy
- [ ] Implement sequencing proxy
- [ ] Plot each metric over time
- [ ] Validate whether the numbers seem physically reasonable
- [ ] Document formulas and definitions

### Done when
- the system outputs 3 metrics for one swing with plots

---

## Phase 5 — Feedback Engine
**Goal:** Turn metrics into explainable statements.

### Tasks
- [ ] Define threshold logic for each metric
- [ ] Write feedback templates
- [ ] Attach numerical evidence to statements
- [ ] Add warnings for low-confidence measurements
- [ ] Review statements for clarity and honesty

### Done when
- one swing video produces short, supported feedback

---

## Phase 6 — Visualization and Demo
**Goal:** Make the project demoable.

### Tasks
- [ ] Build a simple results summary view
- [ ] Save key-frame images
- [ ] Overlay landmarks on important frames
- [ ] Create plots for motion evidence
- [ ] Produce an end-to-end sample output

### Done when
- the project can be shown and explained clearly to another person

---

## Phase 7 — Robustness and Expansion
**Goal:** Make the project less brittle.

### Tasks
- [ ] Test on more swing videos
- [ ] Log failure cases
- [ ] Improve event heuristics
- [ ] Improve smoothing / filtering
- [ ] Refine metric formulas
- [ ] Expand supported scenarios carefully

### Done when
- the system works on multiple examples with fewer obvious failures

---

## Phase 8 — Add a Real ML Layer
**Goal:** Turn the project from motion-analysis MVP into a clearer ML capstone.

### Possible options
- [ ] Train a classifier for one mechanical issue
- [ ] Train a model to detect swing phases from trajectories
- [ ] Cluster swings by movement pattern
- [ ] Use anomaly detection for unusual mechanics
- [ ] Learn a score from extracted features

### Rule
Do not start this phase until the baseline explainable pipeline works.

---

# 11. Risks and Bottlenecks

## Risk 1: Pose noise
Raw landmarks may be unstable or partially wrong.  
**Mitigation:** Use smoothing, confidence filtering, and narrow use cases.

## Risk 2: Camera inconsistency
Different views break assumptions.  
**Mitigation:** Support one view only at first.

## Risk 3: Ambiguous event definitions
“Contact” and “launch” are hard to define precisely.  
**Mitigation:** Use proxies first and document them honestly.

## Risk 4: Overengineering
Trying to build the dream system too early will kill the project.  
**Mitigation:** Freeze MVP scope and finish one narrow slice.

## Risk 5: Fake feedback
It is easy to write feedback that sounds smart but is not measurable.  
**Mitigation:** Require evidence for every claim.

---

# 12. What to Avoid Early

- building custom deep learning models too soon
- supporting multiple views
- real-time processing
- full bat-tracking
- trying to solve all mechanics at once
- writing “coach language” before metrics are solid
- adding drill recommendations before evidence is trustworthy
- making the UI fancy before the analysis works

---

# 13. First Technical Milestone

## Milestone 1
**Given one side-view swing video, produce smoothed pose trajectories and a head-movement analysis with plots and annotated frames.**

### Why this milestone
This is the smallest serious version of the project that proves:
- video ingestion works
- pose extraction works
- motion data can be processed
- a real baseball-relevant metric can be measured
- the system can produce evidence, not just claims

### Tasks
- [ ] Select one clean side-view swing video
- [ ] Extract pose landmarks frame-by-frame
- [ ] Store landmarks in a usable format
- [ ] Smooth keypoint trajectories
- [ ] Normalize using body reference
- [ ] Compute head movement across the swing window
- [ ] Plot head x/y movement over time
- [ ] Save 2–3 annotated key frames
- [ ] Write a short analysis summary

### Done when
- the system can output a believable head movement result for one swing

---

# 14. Weekly Working Style

## Weekly rule
Each week should have:
1. one engineering goal
2. one understanding goal

### Example
**Engineering goal:** implement smoothing  
**Understanding goal:** learn why the chosen smoothing method is appropriate

## Weekly cadence

### Day 1
- define exact task
- define input/output
- write small design notes

### Day 2–4
- implement one module only
- keep scope narrow
- test on one known example

### Day 5
- inspect results visually
- note failures
- ask whether the behavior makes physical sense

### Day 6
- refactor
- document lessons
- clean files and structure

### Day 7
- review progress
- update checklist
- define next target

---

# 15. Definition of Success

## MVP success
The MVP is successful if it can:
- process one swing video reliably
- extract pose trajectories
- compute 3 interpretable metrics
- generate simple evidence-backed feedback
- present visuals that justify the conclusions

## Capstone success
The project is successful if it demonstrates:
- end-to-end systems thinking
- clear scope control
- interpretable feature design
- solid numerical reasoning
- technical ownership of the pipeline
- optional ML extension after a working baseline

---

# 16. Repo To-Do List

## Immediate
- [ ] Add this plan to the repo
- [ ] Create directory structure
- [ ] Write a short README
- [ ] Define initial camera angle
- [ ] Define sample input video source
- [ ] Choose pose library/tool
- [ ] Create first milestone issue/task list

## Near-term
- [ ] Complete Milestone 1
- [ ] Save plots and outputs
- [ ] Start project journal / build log
- [ ] Record design decisions in repo

---

# 17. Open Questions

- [ ] Which single camera angle will be supported first?
- [ ] What pose estimator will be used initially?
- [ ] What body landmarks are most reliable for this use case?
- [ ] What body reference should be used for normalization?
- [ ] What is the first contact proxy definition?
- [ ] Which 3 mechanics are most realistic for the MVP?
- [ ] What sample videos will be used for early testing?

---

# 18. Build Log

## 2026-04-22
### Status
Project started.

### Completed
- [x] Defined project direction
- [x] Defined final vision vs MVP
- [x] Created staged build roadmap
- [x] Identified first technical milestone

### Notes
Day 1 focus is planning, scope control, and defining the first real deliverable.

---

# 19. Progress Log Template

Copy this block for each work session.

## YYYY-MM-DD HH:MM
### Worked on
- 

### Completed
- [ ] 

### Problems hit
- 

### Decisions made
- 

### Next step
- 

### Notes
- 

---

# 20. Guiding Reminder

Do not try to build the whole future system at once.

Build this in order:
1. measurement
2. interpretation
3. feedback
4. ML extension

**Finished and narrow beats ambitious and half-built.**