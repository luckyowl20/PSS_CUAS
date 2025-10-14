#!/usr/bin/env python3
"""
make_video_simple.py
Create a video (MP4) from a folder of images with bounding box centers marked.
Shows current position (red) and future position 2 seconds ahead (blue).
Edit the CONFIG section below and run:  python make_video_simple.py
"""
import glob
import os
import re
from typing import List, Optional, Tuple, Dict

import cv2

# -------------------- CONFIG --------------------
INPUT_DIR   = r"drone_footage/20190925_111757_1_10_frames"  # folder with images
LABEL_DIR   = r"drone_footage/20190925_111757_1_10_label"   # folder with label files
OUTPUT_PATH = r"output.mp4"                                  # output video file
FPS         = 5                                           # frames per second
PATTERN     = "*.jpg"                                        # glob, e.g., "*.jpg" or "*.png"
RECURSIVE   = False                                          # search subfolders too
RESIZE_W: Optional[int] = None                               # set to int (e.g., 1920) or None
RESIZE_H: Optional[int] = None                               # set to int (e.g., 1080) or None
CURRENT_DOT_RADIUS  = 10                                      # radius of current position dot
CURRENT_DOT_COLOR   = (0, 0, 255)                            # BGR: red
FUTURE_DOT_RADIUS   = 10                                      # radius of future position dot
FUTURE_DOT_COLOR    = (255, 0, 0)                            # BGR: blue
DOT_THICKNESS = -1                                           # -1 fills the circle
FUTURE_SECONDS = 1                                         # seconds into the future
# ------------------------------------------------

def natural_key(s: str) -> List:
    """Natural sort key: treats digit runs as integers."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', s)]

def find_images(input_dir: str, pattern: str, recursive: bool) -> List[str]:
    if recursive:
        paths = glob.glob(os.path.join(input_dir, '**', pattern), recursive=True)
    else:
        paths = glob.glob(os.path.join(input_dir, pattern))
    paths = [p for p in paths if os.path.isfile(p)]
    paths.sort(key=natural_key)
    return paths

def compute_resize(new_w: Optional[int], new_h: Optional[int], w: int, h: int) -> Tuple[int, int]:
    if new_w and new_h:
        return new_w, new_h
    if new_w and not new_h:
        ratio = new_w / w
        return new_w, int(round(h * ratio))
    if new_h and not new_w:
        ratio = new_h / h
        return int(round(w * ratio)), new_h
    return w, h

def get_label_path(image_path: str, label_dir: str) -> str:
    """Get corresponding label file path for an image."""
    basename = os.path.basename(image_path)
    name_without_ext = os.path.splitext(basename)[0]
    return os.path.join(label_dir, name_without_ext + ".txt")

def read_labels(label_path: str) -> List[Tuple[int, float, float, float, float]]:
    """Read YOLO format labels: class x_center y_center width height (normalized)."""
    labels = []
    if not os.path.exists(label_path):
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                cls = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                labels.append((cls, x_center, y_center, width, height))
    return labels

def load_all_labels(images: List[str], label_dir: str) -> Dict[int, List[Tuple[int, float, float, float, float]]]:
    """Load all labels for all frames indexed by frame number."""
    all_labels = {}
    for idx, img_path in enumerate(images):
        label_path = get_label_path(img_path, label_dir)
        labels = read_labels(label_path)
        all_labels[idx] = labels
    return all_labels

def draw_centers_with_future(img, current_labels: List[Tuple[int, float, float, float, float]], 
                              future_labels: Optional[List[Tuple[int, float, float, float, float]]]):
    """Draw dots at current and future bounding box centers."""
    img_h, img_w = img.shape[:2]
    
    # Draw current positions (red)
    for cls, x_norm, y_norm, w_norm, h_norm in current_labels:
        x_pixel = int(x_norm * img_w)
        y_pixel = int(y_norm * img_h)
        cv2.circle(img, (x_pixel, y_pixel), CURRENT_DOT_RADIUS, CURRENT_DOT_COLOR, DOT_THICKNESS)
    
    # Draw future positions (blue) if available
    if future_labels:
        for cls, x_norm, y_norm, w_norm, h_norm in future_labels:
            x_pixel = int(x_norm * img_w)
            y_pixel = int(y_norm * img_h)
            cv2.circle(img, (x_pixel, y_pixel), FUTURE_DOT_RADIUS, FUTURE_DOT_COLOR, DOT_THICKNESS)
    
    return img

def main():
    images = find_images(INPUT_DIR, PATTERN, RECURSIVE)
    if not images:
        print(f"[ERROR] No images found with pattern '{PATTERN}' in '{INPUT_DIR}'.")
        return

    # Load all labels first
    print("[INFO] Loading all labels...")
    all_labels = load_all_labels(images, LABEL_DIR)
    
    # Calculate frames for future offset
    frames_ahead = int(FUTURE_SECONDS * FPS)
    print(f"[INFO] Looking {frames_ahead} frames ahead ({FUTURE_SECONDS}s at {FPS} fps)")

    # Read first valid frame to get size
    first_frame = None
    for path in images:
        img = cv2.imread(path)
        if img is not None:
            first_frame = img
            break
    if first_frame is None:
        print("[ERROR] Could not read any images (are they valid files?)")
        return

    h, w = first_frame.shape[:2]
    out_w, out_h = compute_resize(RESIZE_W, RESIZE_H, w, h)

    # Prepare writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (out_w, out_h))
    alt_out = None
    if not writer.isOpened():
        alt_out = os.path.splitext(OUTPUT_PATH)[0] + ".avi"
        print(f"[WARN] mp4v could not be opened. Falling back to MJPG: {alt_out}")
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(alt_out, fourcc, FPS, (out_w, out_h))
        if not writer.isOpened():
            print("[ERROR] Failed to open a video writer for both MP4 and AVI.")
            return

    total = 0
    skipped = 0
    total_boxes = 0
    future_predictions = 0
    
    for idx, path in enumerate(images):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Skipping unreadable: {path}")
            skipped += 1
            continue
        
        # Get current frame labels
        current_labels = all_labels.get(idx, [])
        total_boxes += len(current_labels)
        
        # Get future frame labels (if available)
        future_idx = idx + frames_ahead
        future_labels = None
        if future_idx < len(images):
            future_labels = all_labels.get(future_idx, [])
            if future_labels:
                future_predictions += len(future_labels)
        
        # Draw current and future centers before resizing
        if current_labels or future_labels:
            img = draw_centers_with_future(img, current_labels, future_labels)
        
        # Resize if needed
        if (img.shape[1], img.shape[0]) != (out_w, out_h):
            img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        
        writer.write(img)
        total += 1
        if total % 200 == 0:
            print(f"[INFO] Wrote {total} frames...")

    writer.release()
    out_path = alt_out if alt_out else OUTPUT_PATH
    print(f"[DONE] Wrote {total} frames to '{out_path}'. Skipped: {skipped}.")
    print(f"[INFO] Size: {out_w}x{out_h} @ {FPS} fps.")
    print(f"[INFO] Total current positions (red): {total_boxes}")
    print(f"[INFO] Total future positions (blue): {future_predictions}")

if __name__ == "__main__":
    main()