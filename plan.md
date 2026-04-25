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
- call pose estimator
- extract key landmarks per frame
- collect confidence values
- save raw keypoint output

### Outputs
- per-frame landmark arrays
- confidence arrays
- raw pose data file

---

## Module 3: Landmark Processing
**Purpose:** Turn noisy pose output into usable motion data.

### Responsibilities
- smoothing
- interpolation for small gaps
- normalization
- reference frame handling

### Outputs
- cleaned trajectories
- normalized trajectories

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

---

## Module 5: Feature Extraction
**Purpose:** Compute interpretable mechanics metrics.

### Responsibilities
- calculate motion-based features
- compare body regions over time
- derive timing relationships
- produce metrics with units / normalized units

### Outputs
- feature dictionary
- metric tables
- time-series arrays

---

## Module 6: Feedback Engine
**Purpose:** Turn metrics into evidence-backed feedback.

### Responsibilities
- apply rules to features
- generate statements
- attach numeric evidence
- attach confidence or warning flags

### Outputs
- feedback statements
- explanation text
- issue flags

---

## Module 7: Visualization / Report
**Purpose:** Show proof.

### Responsibilities
- key-frame plots
- trajectory plots
- metric-over-time charts
- simple final report

### Outputs
- images
- plots
- report assets

---

## Module 8: Evaluation
**Purpose:** Verify whether results are stable and believable.

### Responsibilities
- inspect metric consistency
- compare across sample videos
- track failure cases
- document assumptions and limitations

### Outputs
- test notes
- evaluation results
- failure log

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