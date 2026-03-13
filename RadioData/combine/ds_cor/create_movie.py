
from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
import imageio.v2 as imageio
import numpy as np





# =========================
# Helpers
# =========================
# Expected filename: ds_cor_YYYYMMDDTHHMMSS.png
PAT = re.compile(r"^ds_cor_(\d{8})T(\d{6})\.png$")

def parse_ts_from_name(p: Path) -> datetime | None:
    m = PAT.match(p.name)
    if not m:
        return None
    ymd, hms = m.group(1), m.group(2)
    return datetime.strptime(ymd + hms, "%Y%m%d%H%M%S")


def pad_to_shape(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Pad image (H,W,C) or (H,W) to (out_h,out_w,...) with black pixels."""
    if img.ndim == 2:
        h, w = img.shape
        c = None
    else:
        h, w, c = img.shape

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
# =========================
# User settings
# =========================
start_time = "2022-06-13T01:00:00"
end_time   = "2022-06-13T05:01:00"

# Windows path (as given). If you prefer WSL path, you can replace this with:
#   r"/mnt/f/wsl/home/kinno-7010/Research/RadioData/combine/ds_cor"
input_dir = Path(r"F:/wsl/home/kinno-7010/Research/RadioData/combine/ds_cor")

# Output settings
fps = 10  # change if you want faster/slower video
output_mp4 = input_dir / f"ds_cor_{start_time.replace('-','').replace(':','')}_{end_time.replace('-','').replace(':','')}.mp4"
dt_start = datetime.fromisoformat(start_time)
dt_end   = datetime.fromisoformat(end_time)

if not input_dir.exists():
    raise FileNotFoundError(f"Input directory not found: {input_dir}")

# Collect files within time range
items: list[tuple[datetime, Path]] = []
for p in input_dir.glob("ds_cor_*.png"):
    ts = parse_ts_from_name(p)
    if ts is None:
        continue
    if dt_start <= ts <= dt_end:
        items.append((ts, p))

items.sort(key=lambda x: x[0])

if not items:
    raise RuntimeError(f"No PNG files found in the range {dt_start} .. {dt_end}")

# Read once to determine max frame size (robust to mixed sizes)
frames = []
max_h, max_w = 0, 0
for ts, p in items:
    img = imageio.imread(p)
    if img.ndim == 2:
        h, w = img.shape
    else:
        h, w = img.shape[:2]
    max_h = max(max_h, h)
    max_w = max(max_w, w)
    frames.append(img)

# Write MP4
# imageio will use ffmpeg backend; if it complains, install: pip install "imageio[ffmpeg]"
writer = imageio.get_writer(output_mp4, fps=fps, codec="libx264", quality=2)

try:
    for img in frames:
        img2 = pad_to_shape(img, max_h, max_w)

        # Ensure 3-channel RGB for video
        if img2.ndim == 2:
            img2 = np.stack([img2]*3, axis=-1)
        elif img2.shape[2] == 4:
            img2 = img2[:, :, :3]

        writer.append_data(img2)
finally:
    writer.close()

print(f"Saved video: {output_mp4}")
print(f"Frames: {len(items)}, FPS: {fps}")
