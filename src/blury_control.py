import cv2
import os
import glob

def remove_blurry_frames(folder_path: str, threshold: float = 80.0) -> int:
    """
    Removes images with focus scores below the threshold.
    
    Args:
        folder_path: Directory containing extracted frames.
        threshold: Focus threshold. Lower = looser (keeps softer images), 
                   Higher = stricter (only keeps super sharp images).
    """
    image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
    removed_count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Calculate focus measure (variance of Laplacian)
        focus_score = cv2.Laplacian(img, cv2.CV_64F).var()

        if focus_score < threshold:
            os.remove(img_path)
            removed_count += 1

    print(f"[BlurFilter] Removed {removed_count} blurry frames (Focus Score < {threshold}).")
    return removed_count