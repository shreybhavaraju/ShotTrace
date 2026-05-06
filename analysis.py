# Form-similarity analysis. Builds a "good shot" template by averaging keypoint
# sequences across all made shots, then measures how far each individual shot
# deviates from that template. Outputs the drift of that distance over the
# chronological shot order — does my form drift as I get tired?

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

KEYPOINTS_DIR = Path('keypoints')
FEATURES      = Path('trimmed/features.csv')
TEMPLATE_OUT  = Path('trimmed/good_shot_template.npy')
FIGS          = Path('figs')
FIGS.mkdir(exist_ok=True)


def img_num(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


df = pd.read_csv(FEATURES)
# Sort by IMG number so shot_index actually means "chronological order on the phone."
df['img_num']    = df['filename'].apply(img_num)
df = df.sort_values('img_num').reset_index(drop=True)
df['shot_index'] = df.index

# Load keypoints (drop visibility — we only need spatial position for shape comparison)
kps = np.stack([
    np.load(KEYPOINTS_DIR / fn)[:, :, :2] for fn in df['filename']
])  # (N, 30, 9, 2)

make_mask = df['label_int'].values == 1
miss_mask = ~make_mask

# Average all made shots → "good shot" template. Only frame-by-frame mean over the
# made set; we're not doing anything fancier (median etc.) because the noise filter
# in features.py already dropped the outlier shots that would skew the mean.
template = kps[make_mask].mean(axis=0)   # (30, 9, 2)
np.save(TEMPLATE_OUT, template)

# Euclidean distance from each shot's flattened sequence to the flattened template
flat = kps.reshape(len(df), -1)
flat_template = template.reshape(-1)
distances = np.linalg.norm(flat - flat_template, axis=1)

print(f"loaded {len(df)} shots  ({make_mask.sum()} makes, {miss_mask.sum()} misses)")
print(f"template built from {make_mask.sum()} made shots → {TEMPLATE_OUT}")
print()
print("distance to template:")
print(f"  makes:  mean={distances[make_mask].mean():.3f}  std={distances[make_mask].std():.3f}")
print(f"  misses: mean={distances[miss_mask].mean():.3f}  std={distances[miss_mask].std():.3f}")
print()

# Two-panel figure: distribution by label + drift over time
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

axes[0].hist(distances[make_mask], bins=25, alpha=0.6, color='green',
             edgecolor='black', label='make')
axes[0].hist(distances[miss_mask], bins=25, alpha=0.6, color='red',
             edgecolor='black', label='miss')
axes[0].set_xlabel('Euclidean distance to "good shot" template')
axes[0].set_ylabel('count')
axes[0].set_title('Distance to template — makes vs misses')
axes[0].legend()

axes[1].scatter(df['shot_index'][make_mask], distances[make_mask],
                color='green', alpha=0.5, s=18, label='make')
axes[1].scatter(df['shot_index'][miss_mask], distances[miss_mask],
                color='red',   alpha=0.5, s=18, label='miss')
window = 25
rolling = pd.Series(distances).rolling(window, min_periods=5).mean()
axes[1].plot(df['shot_index'], rolling, color='black', linewidth=2,
             label=f'{window}-shot rolling mean')
axes[1].set_xlabel('shot index (chronological)')
axes[1].set_ylabel('distance to template')
axes[1].set_title('Form drift over the dataset')
axes[1].legend()

plt.tight_layout()
plt.savefig(FIGS / 'template_distance_drift.png', dpi=120)
plt.close()

print(f"figure saved to {FIGS}/template_distance_drift.png")
