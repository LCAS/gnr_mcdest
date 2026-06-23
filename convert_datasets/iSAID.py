"""
iSAID density-map generator (Pathlib + visualisation) - with tqdm.
Channel-wise spatial-mask generation added.
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


# --------------------------------------------------------------------
# 1.  Paths - adjust only `ROOT` (train / val / test)
# --------------------------------------------------------------------
ROOT = Path("../datasets/iSAID/val")          # <-- change as needed
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

# --------------------------------------------------------------------
# 2.  Helper functions
# --------------------------------------------------------------------
def resize_img(img: np.ndarray, target_size: int = 1024) -> np.ndarray:
    """Resize only by width; keep aspect ratio."""
    h, w = img.shape[:2]
    scale = target_size / w
    return cv2.resize(img, (target_size, int(h * scale)), interpolation=cv2.INTER_AREA)


def resize_coord(coord: tuple[int, int], orig_shape: tuple[int, int],
                 target_size: int = 1024) -> tuple[int, int]:
    """Scale an (x, y) coordinate to the width-scaled image."""
    h, w = orig_shape
    scale = target_size / w
    return int(coord[0] * scale), int(coord[1] * scale)


def visualize_channel(channel: np.ndarray, out_path: Path) -> None:
    """Save a colour-coded PNG for a single channel (density or mask)."""
    img_vis = 255 * channel / np.max(channel)
    img_vis = img_vis.astype(np.uint8)
    img_col = cv2.applyColorMap(img_vis, cv2.COLORMAP_PLASMA)
    cv2.imwrite(str(out_path), img_col)

# --------------------------------------------------------------------
# 3.  Class list & Gaussian kernels
# --------------------------------------------------------------------
ISAID_CLASSES = ["plane", "ship", "car", "truck"]
CLASS_TO_IDX  = {c: i for i, c in enumerate(ISAID_CLASSES)}
KERNEL_SIZE   = [8, 8, 5, 6]        # per class

# --------------------------------------------------------------------
# 4.  Main loop
# --------------------------------------------------------------------
ann_paths = sorted(ANNOT_DIR.rglob("*.csv"))
start_time = time.time()

for ann_path in tqdm(ann_paths,
                     desc="Generating density maps",
                     unit="file",
                     colour="magenta"):

    ann_name = ann_path.name
    img_name = ann_path.with_suffix(".png").name
    img_path = IMG_DIR / img_name

    # ---- 4.1 Read & resize image ------------------------------------------------
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[WARN]  Image not found: {img_path}")
        continue

    orig_h, orig_w = img.shape[:2]
    img_resized = resize_img(img, 1024)
    resized_h, resized_w = img_resized.shape[:2]

    # ---- 4.2 Build 4-channel point map ------------------------------------------
    kpoint = np.zeros((len(ISAID_CLASSES), resized_h, resized_w), dtype=np.int8)

    with ann_path.open("r") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 6:
                continue
            x, y, w, h = map(float, parts[:4])
            classname = parts[5].strip()
            if classname not in CLASS_TO_IDX:
                continue
            cls_id = CLASS_TO_IDX[classname]
            cx = int(x + w / 2.0)
            cy = int(y + h / 2.0)
            cx, cy = resize_coord((cx, cy), (orig_h, orig_w), 1024)
            cx = np.clip(cx, 0, resized_w - 1)
            cy = np.clip(cy, 0, resized_h - 1)
            kpoint[cls_id, cy, cx] = 1

    # ---- 4.3 Gaussian smoothing per channel ------------------------------------
    density_map = np.empty_like(kpoint, dtype=np.float32)
    for cls in range(len(ISAID_CLASSES)):
        density_map[cls] = gaussian_filter(
            kpoint[cls].astype(np.float32), sigma=KERNEL_SIZE[cls]
        )

    # ---- 4.4 Channel-wise mask binning (simplified) ----------------------------
    distance = 5          # bin width in “density-units”


    dist_maps = np.array(
        [
            cv2.distanceTransform(
                (255 * (1 - kpoint[ch].astype(np.uint8))),  # binary mask 0 = object
                cv2.DIST_L2, distance                               # Euclidean distance
            )
            for ch in range(kpoint.shape[0])
        ],
        dtype=np.float32
    )   # shape (C, H, W)

    
    # `np.digitize` automatically caps at the number of bins.
    bins = np.array([distance, distance*2, distance*3, distance*4, distance*5, distance*6, distance*8, distance*12, distance*18, distance*28])   # 10 bin edges
    # Apply to every channel at once
    spatial_mask = np.digitize(dist_maps, bins, right=False).astype(np.int32)
    # Values >= 140 become 10 (already handled by digitize)
    # The result has shape (4, h, w) with integer classes 0-10.

    if (PRODUCE_VISUALISATIONS):
        # ---- 4.5 Visualise density channels ---------------------------------------
        for cls in range(len(ISAID_CLASSES)):
            ch_vis_path = GT_SHOW_DIR / f"{img_name}_{ISAID_CLASSES[cls]}.jpg"
            visualize_channel(density_map[cls], ch_vis_path)

        # ---- 4.6 Visualise masks (same colour map) --------------------------------
        for cls in range(len(ISAID_CLASSES)):
            mask_vis_path = GT_SHOW_DIR / f"{img_name}_{ISAID_CLASSES[cls]}_mask.jpg"
            visualize_channel(spatial_mask[cls], mask_vis_path)


    # Write resized images
    img_out_path = OUT_IMG_DIR / img_name
    cv2.imwrite(str(img_out_path), img_resized)

    # ---- 4.7 Save both datasets to HDF5 ---------------------------------------
    h5_path = OUT_H5_DIR / ann_name.replace(".csv", ".h5")
    with h5py.File(str(h5_path), "w") as hf:
        hf.create_dataset(
            "density_map",
            data=density_map,
            compression="gzip",
            dtype="float32",
        )
        hf.create_dataset(
            "mask",
            data=spatial_mask,          # 4-channel integer mask
            compression="gzip",
            dtype="int32",
        )

# --------------------------------------------------------------------
# 5.  Summary
# --------------------------------------------------------------------
total = time.time() - start_time
print(f"\n✓  Finished {len(ann_paths)} files in {total:.1f}s")