from src.io.video_loader import load_video_metadata
from src.io.video_loader import extract_frames_with_timestamps
from src.visualization.frame_saver import save_debug_frames

def main():
    video_path = "data/raw/mike_trout_swing_01.mp4"
    metadata = load_video_metadata(video_path)

    print("Video loaded successfully.")
    print(f"Path: {metadata['path']}")
    print(f"FPS: {metadata['fps']:.2f}")
    print(f"Frame count: {metadata['frame_count']}")
    print(f"Resolution: {metadata['width']} x {metadata['height']}")
    print(f"Duration: {metadata['duration']:.2f} seconds")
    
    frames_with_timestamps = extract_frames_with_timestamps(video_path)
    save_debug_frames(frames_with_timestamps, "outputs/debug_frames")

if __name__ == "__main__":
    main()