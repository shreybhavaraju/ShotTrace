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
    # kps shape: (30, 9, 3) — 30 frames, 9 joints, [x, y, visibility]
    angles = np.array([elbow_angle(kps[f]) for f in range(30)])
    wrist_y = kps[:, RWRI, 1]

    # Image y-axis points down, so the highest wrist position is the minimum y.
    peak_wrist_frame = int(wrist_y.argmin())

    return {
        'max_elbow_extension': float(angles.max()),
        'release_frame_est':   int(angles.argmax()),
        'peak_wrist_height':   float(-wrist_y.min()),
        'peak_wrist_frame':    peak_wrist_frame,
        'drive_amplitude':     float(wrist_y[0] - wrist_y.min()),
        'follow_through_x':    float(kps[peak_wrist_frame, RWRI, 0]),
    }


manifest = pd.read_csv(MANIFEST)
print(f"loading {len(manifest)} shots...")

rows = []
for _, row in manifest.iterrows():
    kps = np.load(KEYPOINTS_DIR / row['filename'])
    feats = shot_features(kps)
    feats['filename'] = row['filename']
    feats['label_int'] = int(row['label_int'])
    rows.append(feats)

features = pd.DataFrame(rows)
features = features[['filename', 'label_int',
                     'max_elbow_extension', 'release_frame_est',
                     'peak_wrist_height', 'peak_wrist_frame',
                     'drive_amplitude', 'follow_through_x']]
features.to_csv(OUT, index=False)

print(f"saved {len(features)} rows to {OUT}\n")
print("summary stats:")
print(features.describe().round(3))
print("\nby label (0=miss, 1=make):")
print(features.groupby('label_int').mean(numeric_only=True).round(3))
