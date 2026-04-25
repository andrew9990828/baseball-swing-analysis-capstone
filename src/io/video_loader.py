# Load in library
import cv2 as cv
from pathlib import Path

# Create load_video_metadata function
# Takes in a video_path as a string and outputs a dictionary
# Returns metadata in a dictionary
#
# @params takes string of the video you want to extract the metadata from
def load_video_metadata(video_path: str) -> dict:

    # Create a path object
    path = Path(video_path)

    # Raise an error if the path doesnt exist
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {video_path}")

    # Create a capture object from the video_path
    capture = cv.VideoCapture(str(path))

    # If capture fails to open, raise an error
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Load metadata
    fps = capture.get(cv.CAP_PROP_FPS)
    frame_count = int(capture.get(cv.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Calc duration
    if fps > 0:
        duration = frame_count / fps
    else:
        duration = 0
    
    # Release video object
    capture.release()

    # return a dictionary with all the metadata
    return {
        "path": str(path),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


# Returns a list of dictionaries, each dictionary contains information such as:
#       frame_index:
#       timestamp:
#       frame:
# Important to label each frame so we can understand how to break it up into
# action events. Ex. Load, Swing, Followthrough
#
# @params video_path string of the video being extracted
def extract_frames_with_timestamps(video_path: str) -> list[dict]:

    frame_with_time = []
    metadata = load_video_metadata(video_path)
    frame_count = metadata['frame_count']
    fps = metadata['fps']

    cap = cv.VideoCapture(video_path)


    for i in range(frame_count):

        success, frame = cap.read()

        if not success:
            break

        timestamp = i / fps if fps > 0 else 0
        
        frame_with_time.append(
            {"frame_index": i, "timestamp": timestamp, "frame": frame}
        )

    cap.release()

    return frame_with_time


