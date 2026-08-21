import cv2
import os

class FrameExtractor:
    """Extracts frames from video streams at fixed intervals for dataset creation."""
    
    def __init__(self, video_path: str, output_dir: str, interval: int = 3):
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval = interval
        os.makedirs(self.output_dir, exist_ok=True)

    def extract(self) -> int:
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0
        saved_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % self.interval == 0:
                out_path = os.path.join(self.output_dir, f"frame_{frame_idx:04d}.jpg")
                cv2.imwrite(out_path, frame)
                saved_count += 1
            frame_idx += 1

        cap.release()
        print(f"[FrameExtractor] Extracted {saved_count} frames to '{self.output_dir}'")
        return saved_count