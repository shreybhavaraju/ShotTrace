# Per-shot engineered features for the make/miss models.
# Everything here is release-invariant — computed across all 30 frames and reduced
# with .max()/.argmax(). EDA showed measuring at a fixed frame (frame 5) doesn't work
# because the motion-peak release detector lands mid-extension, not full extension.

import numpy as np
import pandas as pd
from pathlib import Path

KEYPOINTS_DIR = Path('keypoints')
MANIFEST = Path('trimmed/manifest_clean.csv')
OUT = Path('trimmed/features.csv')

# Indices into the (9, 3) keypoint array — right side is the shooting arm
RSHO, RELB, RWRI = 2, 4, 6


def elbow_angle(frame_kps):
    sho = frame_kps[RSHO, :2]
    elb = frame_kps[RELB, :2]
    wri = frame_kps[RWRI, :2]
    v1, v2 = sho - elb, wri - elb
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def shot_features(kps):
    # kps shape: (30, 9, 3) — 30 frames, 9 joints, [x, y, visibility].
    # Mask out frames where the right wrist wasn't reliably detected. MediaPipe stores
    # failed frames as (0, 0, 0), which would otherwise win .argmin() on wrist y and
    # corrupt every release-invariant feature.
    valid = kps[:, RWRI, 2] > 0.3
    if valid.sum() < 5:
        return None  # too little real wrist data to compute features safely

    angles = np.full(30, np.nan)
    for f in range(30):
        if valid[f]:
            angles[f] = elbow_angle(kps[f])

    wrist_y = np.where(valid, kps[:, RWRI, 1], np.nan)
    wrist_x = np.where(valid, kps[:, RWRI, 0], np.nan)

    # Image y-axis points down, so the highest wrist position is the minimum y.
    peak_wrist_frame = int(np.nanargmin(wrist_y))

    # drive_amplitude uses the first *valid* frame as the baseline — frame 0 itself
    # might be a failed-detection frame, in which case its raw 0 isn't a real position.
    first_valid = int(np.argmax(valid))

    return {
        'max_elbow_extension': float(np.nanmax(angles)),
        'release_frame_est':   int(np.nanargmax(angles)),
        'peak_wrist_height':   float(-np.nanmin(wrist_y)),
        'peak_wrist_frame':    peak_wrist_frame,
        'drive_amplitude':     float(wrist_y[first_valid] - np.nanmin(wrist_y)),
        'follow_through_x':    float(wrist_x[peak_wrist_frame]),
    }


manifest = pd.read_csv(MANIFEST)
print(f"loading {len(manifest)} shots...")

rows = []
skipped = 0
for _, row in manifest.iterrows():
    kps = np.load(KEYPOINTS_DIR / row['filename'])
    feats = shot_features(kps)
    if feats is None:
        skipped += 1
        continue
    feats['filename'] = row['filename']
    feats['label_int'] = int(row['label_int'])
    rows.append(feats)

if skipped:
    print(f"skipped {skipped} shots with too few valid wrist frames")

features = pd.DataFrame(rows)
features = features[['filename', 'label_int',
                     'max_elbow_extension', 'release_frame_est',
                     'peak_wrist_height', 'peak_wrist_frame',
                     'drive_amplitude', 'follow_through_x']]

# Drop shots with truly impossible feature magnitudes — these are normalization
# blow-ups (MediaPipe measured shoulder width as ~0 on a rotated/occluded shot, so
# every coordinate divides into a huge number). A 20-shoulder-width displacement is
# physically impossible; values below that may be noisy but could still be real
# signal from far-camera shots, so we keep them and let the models handle the noise.
plausible = (features['drive_amplitude'] <= 20) & (features['follow_through_x'].abs() <= 20)
n_dropped = (~plausible).sum()
features = features[plausible].reset_index(drop=True)
if n_dropped:
    print(f"dropped {n_dropped} shots with broken normalization (extreme magnitudes)")

features.to_csv(OUT, index=False)

print(f"saved {len(features)} rows to {OUT}\n")
print("summary stats:")
print(features.describe().round(3))
print("\nby label (0=miss, 1=make):")
print(features.groupby('label_int').mean(numeric_only=True).round(3))
