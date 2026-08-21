from frame_extract import FrameExtractor
from SSIM_filter import SSIMFilter
from blury_control import remove_blurry_frames


video_file = "sample.mp4"
output_folder = "./raw_frames"

# ------------------------------------------
# Step 1: Extract Raw Frames
# ------------------------------------------
print("--- Step 1: Frame Extraction ---")
extractor = FrameExtractor(video_path=video_file, output_dir=output_folder, interval=3)
total_extracted = extractor.extract()

# ------------------------------------------
# Step 2: Remove Blurry Frames (BEFORE SSIM)
# ------------------------------------------
print("\n--- Step 2: Blur Filtering ---")
# threshold=80.0 is a solid default.
# Increase to 100+ for stricter sharpness; lower to 50-60 if lighting is low.
blurry_removed = remove_blurry_frames(folder_path=output_folder, threshold=80.0)

# ------------------------------------------
# Step 3: Remove Duplicate Frames (SSIM)
# ------------------------------------------
print("\n--- Step 3: SSIM Deduplication ---")
filter_tool = SSIMFilter(
    folder_path=output_folder,
    threshold=0.95,        # Delete frame if >= 95% similar to last kept frame
    resize_dim=(320, 320)  # Speeds up SSIM computation
)
duplicates_removed = filter_tool.filter_duplicates()

# ------------------------------------------
# Pipeline Summary
# ------------------------------------------
remaining = total_extracted - blurry_removed - duplicates_removed
print(f"\n==========================================")
print(f"PIPELINE COMPLETE")
print(f"Total Extracted : {total_extracted}")
print(f"Blurry Discarded: {blurry_removed}")
print(f"Static Discarded: {duplicates_removed}")
print(f"Final Clean Set : {remaining} frames ready in '{output_folder}'")
print(f"==========================================")