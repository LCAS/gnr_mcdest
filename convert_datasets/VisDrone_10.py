#!/usr/bin/env python3
"""
VisDrone density‑map generator (Pathlib + visualisation) – adapted to:
  • 5th column contains a class **index** (int)
  • “ignored‑regions” are filled with black
  • No density channel for “others”
  • Produces **10** density & mask channels
"""

import time
from pathlib import Path

import cv2
import h5py
import numpy as np
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# ------------------------------------------------------------------
# 0.  Optional visualisation flag
# ------------------------------------------------------------------
PRODUCE_VISUALISATIONS = False

# ------------------------------------------------------------------
# 1.  Paths – adjust only `ROOT` (train / val / test)
# ------------------------------------------------------------------
ROOT = Path("../datasets/visdrone/val")          # <-- change as needed
ANNOT_DIR = ROOT / "annotations"
IMG_DIR   = ROOT / "images"

# Output
OUT_ROOT    = ROOT.with_name(ROOT.name + "_den")
OUT_IMG_DIR = OUT_ROOT / "images"
OUT_H5_DIR  = OUT_ROOT / "gt_density_map"
GT_SHOW_DIR = OUT_ROOT / "gt_show"

# Create output dirs
for d in (OUT_IMG_DIR, OUT_H5_DIR, GT_SHOW_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 2.  Helper functions
# ------------------------------------------------------------------
def resize_img(img: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """Resize the image so that its width becomes `target_size`."""
    h, w = img.shape[:2]
    scale = target_size / w
    return cv2.resize(img, (target_size, int(h * scale)), interpolation=cv2.INTER_AREA)

def resize_coord(coord: tuple[int, int], orig_shape: tuple[int, int],
                 target_size: int = 1024) -> tuple[int, int]:
    """Scale an (x, y) coordinate to the width‑scaled image."""
    h, w = orig_shape
    scale = target_size / w
    return int(coord[0] * scale), int(coord[1] * scale)

def visualize_channel(channel: np.ndarray, out_path: Path) -> None:
    """Save a colour‑coded jpg for a single channel (density or mask)."""
    # Normalise to 0‑255, cast to uint8, apply plasma cmap
    img_vis = 255 * channel / np.max(channel) if np.max(channel) > 0 else channel
    img_vis = img_vis.astype(np.uint8)
    img_col = cv2.applyColorMap(img_vis, cv2.COLORMAP_PLASMA)
    cv2.imwrite(str(out_path), img_col)

# ------------------------------------------------------------------
# 3.  Class list & Gaussian kernels
# ------------------------------------------------------------------
# The order is the same as the original “VisDrone_category_buf”:
#   0 = ignored‑regions (no density channel)
#   1 = pedestrian
#   2 = people
#   3 = bicycle
#   4 = car
#   5 = van
#   6 = truck
#   7 = tricycle
#   8 = awning‑tricycle
#   9 = bus
#   10 = motor
#   11 = others   (no density channel)
CLASSES = [
    "ignored-regions",
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
    "others",
]
# We keep a mapping from the 12‑class index to the density‑channel index
# (0 and 11 are skipped, so they map to None)
INDEX_TO_DENSITY_CH = {i: (i-1 if i>=1 and i<11 else None) for i in range(12)}
# e.g. INDEX_TO_DENSITY_CH[3] == 2  (bicycle → channel 2)
KERNEL_SIZE   = [4, 4, 4, 8, 8, 8, 6, 6, 8, 8]        # per class



# ------------------------------------------------------------------
# 4.  Main loop
# ------------------------------------------------------------------
ann_paths = sorted(ANNOT_DIR.rglob("*.txt"))
start_time = time.time()

for ann_path in tqdm(ann_paths, desc="Generating density maps", unit="file", colour="magenta"):
    ann_name = ann_path.name
    img_name = ann_path.with_suffix(".jpg").name
    img_path = IMG_DIR / img_name

    # ---- 4.1 Read & resize image ---------------------------------
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN]  Image not found: {img_path}")
        continue

    orig_h, orig_w = img.shape[:2]
    img_resized = resize_img(img, 1024)
    resized_h, resized_w = img_resized.shape[:2]

    # ---- 4.2 Build 10‑channel point map --------------------------
    # We build a 12‑channel point map first, then strip the unused ones.
    kpoint_all = np.zeros((len(CLASSES), resized_h, resized_w), dtype=np.int8)

    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 5:      # at least 5 fields: bbox + class index
                continue
            # bbox
            x, y, w, h = map(float, parts[:4])
            # class index (5th column)
            cls_id = int(parts[5])
            if cls_id < 0 or cls_id >= len(CLASSES):
                continue
            # centre of bbox
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            cx, cy = resize_coord((cx, cy), (orig_h, orig_w), 1024)
            cx = np.clip(cx, 0, resized_w - 1)
            cy = np.clip(cy, 0, resized_h - 1)
            kpoint_all[cls_id, cy, cx] = 1

            # Special handling for ignored‑regions: fill black
            if cls_id == 0:   # ignored‑regions
                # draw filled black rectangle on the resized image
                bb_resized = (
                    int(x * 1024 / orig_w),          # x1
                    int(y * 1024 / orig_w),          # y1
                    int((x + w) * 1024 / orig_w),    # x2
                    int((y + h) * 1024 / orig_w),    # y2
                )
                cv2.rectangle(
                    img_resized,
                    (bb_resized[0], bb_resized[1]),
                    (bb_resized[2], bb_resized[3]),
                    color=(0, 0, 0),
                    thickness=-1,
                )

    # ---- 4.3 Keep only the 10 density channels (skip 0 & 11) ------
    density_map = np.zeros((10, resized_h, resized_w), dtype=np.float32)
    for src_idx, dst_idx in INDEX_TO_DENSITY_CH.items():
        if dst_idx is None:
            continue
        # Gaussian smoothing – you can adjust the kernel size per class
        density_map[dst_idx] = gaussian_filter(
            kpoint_all[src_idx].astype(np.float32), sigma=KERNEL_SIZE[dst_idx]
        )

    # ---- 4.4 Channel-wise mask binning (simplified) ----------------------------
    distance = 5          # bin width in “density-units”

    # We can remap like this so we don't need to refer to a reverse mapping
    kpoint = kpoint_all[1:11]

    dist_maps = np.array(
        [
            cv2.distanceTransform(
                (255 * (1 - kpoint[ch].astype(np.uint8))),  # binary mask 0 = object
                cv2.DIST_L2, distance                               # Euclidean distance
            )
            for ch in range(density_map.shape[0])
        ],
        dtype=np.float32
    )   # shape (C, H, W)

    
    # `np.digitize` automatically caps at the number of bins.
    bins = np.array([distance, distance*2, distance*3, distance*4, distance*5, distance*6, distance*8, distance*12, distance*18, distance*28])   # 10 bin edges
    # Apply to every channel at once
    spatial_mask = np.digitize(dist_maps, bins, right=False).astype(np.int32)

    if PRODUCE_VISUALISATIONS:
        # ---- 4.5 Visualise density channels --------------------------
        for ch in range(10):
            ch_vis_path = GT_SHOW_DIR / f"{img_name}_density_{CLASSES[1:11][ch]}.jpg"
            visualize_channel(density_map[ch], ch_vis_path)

        # ---- 4.6 Visualise mask channels ----------------------------
        for ch in range(10):
            mask_vis_path = GT_SHOW_DIR / f"{img_name}_mask_{CLASSES[1:11][ch]}.jpg"
            visualize_channel(spatial_mask[ch], mask_vis_path)

    # ---- 4.7 Save the image (with black boxes) --------------------
    img_out_path = OUT_IMG_DIR / img_name
    cv2.imwrite(str(img_out_path), img_resized)

    # ---- 4.8 Save both datasets to HDF5 --------------------------
    h5_path = OUT_H5_DIR / ann_name.replace(".txt", ".h5")
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset(
            "density_map",
            data=density_map,
            compression="gzip",
            dtype="float32",
        )
        hf.create_dataset(
            "mask",
            data=spatial_mask,
            compression="gzip",
            dtype="int32",
        )

# ------------------------------------------------------------------
# 5.  Summary
# ------------------------------------------------------------------
total = time.time() - start_time
print(f"\n✓  Finished {len(ann_paths)} files in {total:.1f}s")