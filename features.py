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

# Indices into the (9, 3) keypoint array. Right side is the shooting arm; left is the
# guide hand (which should be pretty quiet in a clean one-handed shot).
NOSE = 0
LSHO, RSHO = 1, 2
LELB, RELB = 3, 4
LWRI, RWRI = 5, 6


def elbow_angle(frame_kps, sho_idx, elb_idx, wri_idx):
    sho = frame_kps[sho_idx, :2]
    elb = frame_kps[elb_idx, :2]
    wri = frame_kps[wri_idx, :2]
    v1, v2 = sho - elb, wri - elb
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def shot_features(kps):
    # kps shape: (30, 9, 3) — 30 frames, 9 joints, [x, y, visibility].
    # Mask frames where the right wrist failed (MediaPipe stores those as (0,0,0),
    # which would corrupt every reduce-over-time feature).
    valid = kps[:, RWRI, 2] > 0.3
    if valid.sum() < 5:
        return None

    # Angles per frame for both arms (NaN where wrist invalid)
    r_angles = np.full(30, np.nan)
    l_angles = np.full(30, np.nan)
    for f in range(30):
        if valid[f]:
            r_angles[f] = elbow_angle(kps[f], RSHO, RELB, RWRI)
        # Left elbow uses its own visibility — we want left-arm signal whenever it's
        # there, regardless of right-wrist validity, so check left wrist independently.
        if kps[f, LWRI, 2] > 0.3:
            l_angles[f] = elbow_angle(kps[f], LSHO, LELB, LWRI)

    rwx = np.where(valid, kps[:, RWRI, 0], np.nan)
    rwy = np.where(valid, kps[:, RWRI, 1], np.nan)

    peak_wrist_frame   = int(np.nanargmin(rwy))
    release_frame_est  = int(np.nanargmax(r_angles))
    first_valid        = int(np.argmax(valid))

    # Frame-to-frame velocity of the right wrist. Skip NaN-bordering pairs so we don't
    # fabricate a velocity from a failed frame.
    wrist_pos = np.where(valid[:, None], kps[:, RWRI, :2], np.nan)
    wrist_vel = np.linalg.norm(np.diff(wrist_pos, axis=0), axis=1)   # (29,)

    # Per-frame velocity of the elbow angle (degrees per frame). Captures arm-whip speed.
    angle_vel = np.diff(r_angles)

    # Median nose y across valid frames — proxy for head/face position. Used as a
    # head-relative reference for wrist height (above-head vs. below-head).
    nose_visible = kps[:, NOSE, 2] > 0.3
    nose_y_ref = float(np.nanmedian(kps[nose_visible, NOSE, 1])) if nose_visible.any() else np.nan

    # Shoulder tilt at peak: angle of the line connecting the shoulders, relative to
    # horizontal. A real shot rotates the shooting shoulder up.
    if kps[peak_wrist_frame, LSHO, 2] > 0.3 and kps[peak_wrist_frame, RSHO, 2] > 0.3:
        ls = kps[peak_wrist_frame, LSHO, :2]
        rs = kps[peak_wrist_frame, RSHO, :2]
        shoulder_tilt = float(np.degrees(np.arctan2(rs[1] - ls[1], rs[0] - ls[0])))
    else:
        shoulder_tilt = np.nan

    return {
        # Original 6 — same definitions, kept for continuity.
        'max_elbow_extension': float(np.nanmax(r_angles)),
        'release_frame_est':   release_frame_est,
        'peak_wrist_height':   float(-np.nanmin(rwy)),
        'peak_wrist_frame':    peak_wrist_frame,
        'drive_amplitude':     float(rwy[first_valid] - np.nanmin(rwy)),
        'follow_through_x':    float(rwx[peak_wrist_frame]),

        # New 8 — guide-hand asymmetry, motion speed, posture, timing.
        'left_max_elbow_ext':       float(np.nanmax(l_angles)) if not np.isnan(l_angles).all() else np.nan,
        'guide_hand_asymmetry':     float(np.nanmax(r_angles) - np.nanmax(l_angles)) if not np.isnan(l_angles).all() else np.nan,
        'max_wrist_velocity':       float(np.nanmax(wrist_vel)) if not np.isnan(wrist_vel).all() else 0.0,
        'max_arm_whip_speed':       float(np.nanmax(np.abs(angle_vel))) if not np.isnan(angle_vel).all() else 0.0,
        'release_to_peak_lag':      peak_wrist_frame - release_frame_est,
        'wrist_above_head':         float(-np.nanmin(rwy) - (-nose_y_ref)) if not np.isnan(nose_y_ref) else np.nan,
        'shoulder_tilt_at_peak':    shoulder_tilt,
        'post_release_drop':        float(rwy[-1] - np.nanmin(rwy)) if valid[-1] else np.nan,
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
ordered_cols = ['filename', 'label_int',
                'max_elbow_extension', 'release_frame_est',
                'peak_wrist_height', 'peak_wrist_frame',
                'drive_amplitude', 'follow_through_x',
                'left_max_elbow_ext', 'guide_hand_asymmetry',
                'max_wrist_velocity', 'max_arm_whip_speed',
                'release_to_peak_lag', 'wrist_above_head',
                'shoulder_tilt_at_peak', 'post_release_drop']
features = features[ordered_cols]
# Drop any rows where one of the new features came back NaN (rare — only when a key
# joint was completely invisible). Keeps the model from having to special-case missing.
features = features.dropna().reset_index(drop=True)

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
