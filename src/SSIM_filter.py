import os
import glob
import cv2
from skimage.metrics import structural_similarity as ssim

class SSIMFilter:
    """Filters out duplicate or highly similar static frames from an image folder."""

    def __init__(self, folder_path: str, threshold: float = 0.95, resize_dim: tuple = (320, 320)):
        """
        Args:
            folder_path: Path to raw frames folder.
            threshold: SSIM similarity threshold (0.95 = 95% similar).
            resize_dim: Resize dimension for faster SSIM calculation.
        """
        self.folder_path = folder_path
        self.threshold = threshold
        self.resize_dim = resize_dim

    def filter_duplicates(self) -> int:
        image_paths = sorted(glob.glob(os.path.join(self.folder_path, "*.jpg")))
        if not image_paths:
            print(f"[SSIMFilter] No images found in '{self.folder_path}'")
            return 0

        removed_count = 0
        
        # Load the first reference frame
        last_kept_path = image_paths[0]
        last_kept_img = cv2.imread(last_kept_path, cv2.IMREAD_GRAYSCALE)
        if self.resize_dim:
            last_kept_img = cv2.resize(last_kept_img, self.resize_dim)

        for current_path in image_paths[1:]:
            current_img = cv2.imread(current_path, cv2.IMREAD_GRAYSCALE)
            if current_img is None:
                continue

            # Resize current frame for speed comparison
            cmp_current = cv2.resize(current_img, self.resize_dim) if self.resize_dim else current_img

            # Compute SSIM score between last kept frame and current frame
            score, _ = ssim(last_kept_img, cmp_current, full=True)

            if score >= self.threshold:
                # Frame is static/duplicate -> Delete it
                os.remove(current_path)
                removed_count += 1
            else:
                # Significant movement detected -> Keep frame & update reference
                last_kept_img = cmp_current

        print(f"[SSIMFilter] Filtered out {removed_count} duplicate frames (SSIM >= {self.threshold}).")
        print(f"[SSIMFilter] Remaining frames: {len(image_paths) - removed_count}")
        return removed_count