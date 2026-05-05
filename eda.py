# EDA for ShotTrace. Filters bad shots, plots trajectories and elbow angles,
# breaks down make rate by session. Saves figures to figs/ and writes
# manifest_clean.csv (with elbow_angle_release added) to trimmed/.

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KEYPOINTS_DIR = Path('keypoints')
MANIFEST = Path('trimmed/manifest.csv')
FIGS_DIR = Path('figs')
FIGS_DIR.mkdir(exist_ok=True)

RELEASE_FRAME = 5  # 5 frames before release + 25 after
NOSE, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP = range(9)


def img_num(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


def elbow_angle(frame_kps):
    sho = frame_kps[RSHO, :2]
    elb = frame_kps[RELB, :2]
    wri = frame_kps[RWRI, :2]
    v1, v2 = sho - elb, wri - elb
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


# Load manifest and stack every keypoint sequence into one (N, 30, 9, 3) array
manifest = pd.read_csv(MANIFEST)
all_kps = np.stack([np.load(KEYPOINTS_DIR / f) for f in manifest['filename']])
print(f"loaded {len(manifest)} shots, kps shape = {all_kps.shape}")


# 1. Filter bad shots by mean visibility on the shooting arm.
# If MediaPipe can't see the right shoulder/elbow/wrist clearly, the trajectory
# is unusable for downstream features so the whole shot has to go.
shooting_arm = [RSHO, RELB, RWRI]
manifest['arm_visibility'] = all_kps[:, :, shooting_arm, 2].mean(axis=(1, 2))

plt.figure(figsize=(8, 4))
plt.hist(manifest['arm_visibility'], bins=40, edgecolor='black')
plt.axvline(0.5, color='red', linestyle='--', label='threshold=0.5')
plt.xlabel('Mean shooting-arm visibility (over 30 frames)')
plt.ylabel('Number of shots')
plt.title('Pose detection quality per shot')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / '01_visibility_histogram.png', dpi=120)
plt.close()

VIS_THRESHOLD = 0.5  # tuned to drop ~32 shots, matching manual inspection
manifest['good'] = manifest['arm_visibility'] >= VIS_THRESHOLD
print(f"good={manifest['good'].sum()}  bad={(~manifest['good']).sum()}")

manifest_clean = manifest[manifest['good']].reset_index(drop=True)
kps_clean = all_kps[manifest['good'].values]


# 2. Make/miss balance over chronological shot order.
# IMG numbers are assigned in capture order, so sorting by them puts shots
# in the order they were taken — that's what "shot index" should mean here.
manifest_clean['img_num'] = manifest_clean['source_video'].apply(img_num)
manifest_clean = manifest_clean.sort_values('img_num').reset_index(drop=True)
manifest_clean['shot_index'] = manifest_clean.index
kps_clean = kps_clean[manifest_clean.index.values]  # keep arrays in sync after sort

window = 25
rolling = manifest_clean['label_int'].rolling(window, min_periods=5).mean()

plt.figure(figsize=(10, 4))
plt.plot(manifest_clean['shot_index'], rolling, label=f'{window}-shot rolling make rate')
plt.axhline(manifest_clean['label_int'].mean(), color='gray', linestyle='--',
            label=f'overall = {manifest_clean["label_int"].mean():.2f}')
plt.xlabel('Shot index (chronological)')
plt.ylabel('Make rate')
plt.ylim(0, 1)
plt.title('Rolling make rate over the dataset')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / '02_rolling_make_rate.png', dpi=120)
plt.close()


# 3. Right wrist & elbow trajectories overlaid — makes vs misses.
# y is flipped so up on the screen is up on the body (image y goes down).
make_mask = manifest_clean['label_int'].values == 1
miss_mask = ~make_mask

def plot_traj(ax, kps, joint, color, label):
    for shot in kps:
        ax.plot(shot[:, joint, 0], -shot[:, joint, 1], color=color, alpha=0.05)
    mean = kps[:, :, joint, :2].mean(axis=0)
    ax.plot(mean[:, 0], -mean[:, 1], color=color, linewidth=2.5, label=f'{label} (mean)')

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, joint, name in zip(axes, [RWRI, RELB], ['Right wrist', 'Right elbow']):
    plot_traj(ax, kps_clean[make_mask], joint, 'green', 'make')
    plot_traj(ax, kps_clean[miss_mask], joint, 'red', 'miss')
    ax.set_title(f'{name} trajectory  (frames 0→29)')
    ax.set_xlabel('x  (shoulder widths from midpoint)')
    ax.set_ylabel('y  (up positive)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / '03_trajectories.png', dpi=120)
plt.close()


# 4. Elbow angle at release. Coaches cue full extension (~170°) at release;
# any consistent shift between makes and misses is a candidate feature.
release_kps = kps_clean[:, RELEASE_FRAME]
angles = np.array([elbow_angle(f) for f in release_kps])
manifest_clean['elbow_angle_release'] = angles

print(f"makes  : mean={angles[make_mask].mean():.1f}°  std={angles[make_mask].std():.1f}°")
print(f"misses : mean={angles[miss_mask].mean():.1f}°  std={angles[miss_mask].std():.1f}°")

plt.figure(figsize=(8, 4))
plt.hist(angles[make_mask], bins=30, alpha=0.6, color='green', label='make', edgecolor='black')
plt.hist(angles[miss_mask], bins=30, alpha=0.6, color='red', label='miss', edgecolor='black')
plt.xlabel('Elbow angle at release (°)')
plt.ylabel('Count')
plt.title('Right elbow angle at release frame')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / '04_elbow_angle.png', dpi=120)
plt.close()


# 5. Make rate by session. No explicit session field, but each session was
# filmed in one continuous run, so a big jump in IMG numbering means a new day.
GAP_THRESHOLD = 20  # leaves room for skipped clips within a session
diffs = manifest_clean['img_num'].diff().fillna(0)
manifest_clean['session'] = (diffs > GAP_THRESHOLD).cumsum()

session_stats = manifest_clean.groupby('session').agg(
    shots=('label_int', 'count'),
    makes=('label_int', 'sum'),
    make_rate=('label_int', 'mean'),
    first_img=('img_num', 'min'),
    last_img=('img_num', 'max'),
).reset_index()
print('\n' + session_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(session_stats['session'], session_stats['make_rate'],
       color='steelblue', edgecolor='black')
ax.axhline(manifest_clean['label_int'].mean(), color='gray', linestyle='--',
           label=f'overall = {manifest_clean["label_int"].mean():.2f}')
for _, row in session_stats.iterrows():
    ax.text(row['session'], row['make_rate'] + 0.02, f"n={row['shots']}",
            ha='center', fontsize=9)
ax.set_xlabel('Session')
ax.set_ylabel('Make rate')
ax.set_ylim(0, 1)
ax.set_title('Make rate by session')
ax.legend()
plt.tight_layout()
plt.savefig(FIGS_DIR / '05_make_rate_by_session.png', dpi=120)
plt.close()


# Save cleaned manifest with engineered columns for later steps
manifest_clean.to_csv('trimmed/manifest_clean.csv', index=False)
print(f"\nsaved manifest_clean.csv ({len(manifest_clean)} rows)")
print(f"figures saved to {FIGS_DIR}/")
