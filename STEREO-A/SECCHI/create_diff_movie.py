from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
import imageio.v2 as imageio
import numpy as np


# =========================
# User settings (same as before)
# =========================



# =========================
# Helpers
# =========================
# Expected filename: aiaRGB_diff_YYYYMMDD_HHMM.png
PAT = re.compile(r"^aiaRGB_diff_(\d{8})_(\d{4})\.png$")

def parse_ts_from_name(p: Path) -> datetime | None:
    m = PAT.match(p.name)
    if not m:
        return None
    ymd, hm = m.group(1), m.group(2)
    # HHMM -> seconds assumed 00
    return datetime.strptime(ymd + hm, "%Y%m%d%H%M")


def pad_to_shape(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pad image (H,W,C) or (H,W) to (out_h,out_w,...) with black pixels."""
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]

    pad_h = max(0, out_h - h)
    pad_w = max(0, out_w - w)
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if img.ndim == 2:
        return np.pad(img, ((top, bottom), (left, right)), mode="constant", constant_values=0)
    else:
        return np.pad(img, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=0)


# =========================
# Main
# =========================
start_time = "2022-06-13T00:00:00"
end_time   = "2022-06-13T03:30:00"

input_dir = Path(r"/mnt/d/wsl/home/kinno-7010/Research_data/SDO\AIA\output\diff_rgb")

fps = 10
output_mp4 = input_dir / f"aiaRGB_diff_{start_time.replace('-','').replace(':','')}_{end_time.replace('-','').replace(':','')}.mp4"

dt_start = datetime.fromisoformat(start_time)
dt_end   = datetime.fromisoformat(end_time)

if not input_dir.exists():
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

# Collect files within time range
items: list[tuple[datetime, Path]] = []
for p in input_dir.glob("aiaRGB_diff_*.png"):
    ts = parse_ts_from_name(p)
    if ts is None:
        continue
    if dt_start <= ts <= dt_end:
        items.append((ts, p))

items.sort(key=lambda x: x[0])

if not items:
    raise RuntimeError(f"No PNG files found in the range {dt_start} .. {dt_end}")

# Read frames and determine maximum size
frames: list[np.ndarray] = []
max_h, max_w = 0, 0
for ts, p in items:
    img = imageio.imread(p)
    h, w = img.shape[:2] if img.ndim >= 2 else (0, 0)
    max_h = max(max_h, h)
    max_w = max(max_w, w)
    frames.append(img)

# Pad output size to multiples of 16 to avoid ffmpeg macroblock warning
out_h = ((max_h + 15) // 16) * 16
out_w = ((max_w + 15) // 16) * 16

writer = imageio.get_writer(
    output_mp4,
    fps=fps,
    codec="libx264",
    quality=8,  # same idea as previous; you can switch to ffmpeg_params with -crf if preferred
)

try:
    for img in frames:
        img2 = pad_to_shape(img, out_h, out_w)

        # Ensure RGB (3ch)
        if img2.ndim == 2:
            img2 = np.stack([img2] * 3, axis=-1)
        elif img2.shape[2] == 4:
            img2 = img2[:, :, :3]

        writer.append_data(img2)
finally:
    writer.close()

print(f"Saved video: {output_mp4}")
print(f"Frames: {len(items)}, FPS: {fps}, Output size: {out_w}x{out_h}")
