from pathlib import Path

from src.events.event_pipeline import run_event_detection_pipeline


def main():
    input_path = Path("data/processed/pose/mike_trout_swing_01_pose_cleaned.npy")
    output_path = Path("data/processed/events/mike_trout_swing_01_events.json")

    run_event_detection_pipeline(
        input_path=input_path,
        output_path=output_path,
        movement_threshold_ratio=0.25,
        contact_offset=3
    )


if __name__ == "__main__":
    main()