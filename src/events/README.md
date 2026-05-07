# Events Module

This folder contains Module 4: Event / Phase Detection.

## Purpose

Detect rough swing event frames from cleaned pose landmark data.

The event detector uses wrist movement as a v1 swing signal because the current system does not track the bat or ball directly.

## Files

- `event_detector.py`
  - defines the `SwingEvents` dataclass
  - defines the `SwingEventDetector` class
  - calculates wrist speed
  - detects movement start
  - detects peak hand speed
  - estimates a contact proxy

- `event_pipeline.py`
  - loads cleaned landmark data from Module 3
  - runs the event detector
  - saves detected events to a JSON file

## Current Events Detected

- `movement_start`
- `peak_hand_speed`
- `contact_proxy`

## Engineering Decisions

- Wrist speed is used as the first v1 swing movement signal.
- The max speed between left and right wrist is used so handedness does not need to be known yet.
- Contact is only a proxy because bat/ball contact is not currently tracked.
- Event detection is intentionally simple so later modules can build from a working baseline.

## Status

Module 4 is complete for v1.