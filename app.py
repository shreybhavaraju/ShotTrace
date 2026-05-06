# Streamlit dashboard. Static viewer over the 222 cleaned shots — pick any shot,
# scrub through frames to see the pose stick figure, the engineered features,
# distance to the made-shot template, and the CNN's make probability.

import re
import numpy as np
import pandas as pd
import streamlit as st
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from model import ShotCNN, realign_to_release, RELEASE_TARGET_FRAME

KEYPOINTS_DIR = Path('keypoints')
FEATURES      = Path('trimmed/features.csv')
TEMPLATE_PATH = Path('trimmed/good_shot_template.npy')
MODEL_PATH    = Path('model.pt')

# Joint indices into the (9, 3) keypoint array
NOSE = 0
LSHO, RSHO = 1, 2
LELB, RELB = 3, 4
LWRI, RWRI = 5, 6
LHIP, RHIP = 7, 8

# Pairs of joints connected by line segments in the stick figure
SKELETON = [
    (NOSE, LSHO), (NOSE, RSHO),
    (LSHO, RSHO),
    (LSHO, LELB), (LELB, LWRI),
    (RSHO, RELB), (RELB, RWRI),
    (LSHO, LHIP), (RSHO, RHIP),
    (LHIP, RHIP),
]

FEATURE_COLS = [
    'max_elbow_extension', 'release_frame_est',
    'peak_wrist_height',   'peak_wrist_frame',
    'drive_amplitude',     'follow_through_x',
    'left_max_elbow_ext',  'guide_hand_asymmetry',
    'max_wrist_velocity',  'max_arm_whip_speed',
    'release_to_peak_lag', 'wrist_above_head',
    'shoulder_tilt_at_peak', 'post_release_drop',
]


def img_num(name):
    m = re.search(r'(\d+)', name)
    return int(m.group(1)) if m else -1


@st.cache_data
def load_shots():
    df = pd.read_csv(FEATURES)
    df['img_num'] = df['filename'].apply(img_num)
    df = df.sort_values('img_num').reset_index(drop=True)
    keypoints = np.stack([np.load(KEYPOINTS_DIR / fn) for fn in df['filename']])
    template = np.load(TEMPLATE_PATH)
    return df, keypoints, template


@st.cache_resource
def load_model():
    m = ShotCNN()
    m.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    m.eval()
    return m


def predict_make_probability(model, kps, release_frame_est):
    # Replicate model.py's preprocessing exactly: drop visibility, flatten 9x2,
    # add velocity channels, re-align so peak elbow extension lands at frame 15.
    positions  = kps[:, :, :2].reshape(1, 30, 18)
    velocities = np.diff(positions, axis=1, prepend=positions[:, :1, :])
    X = np.concatenate([positions, velocities], axis=2)
    X = realign_to_release(X, [release_frame_est])
    with torch.no_grad():
        logit = model(torch.tensor(X, dtype=torch.float32))
    return float(torch.sigmoid(logit).item())


def plot_skeleton(ax, kps_xy, valid, color, alpha=1.0, linewidth=2.5):
    for a, b in SKELETON:
        if valid[a] and valid[b]:
            ax.plot([kps_xy[a, 0], kps_xy[b, 0]],
                    [-kps_xy[a, 1], -kps_xy[b, 1]],
                    color=color, linewidth=linewidth, alpha=alpha)
    for j in range(9):
        if valid[j]:
            ax.scatter(kps_xy[j, 0], -kps_xy[j, 1],
                       color=color, s=35, zorder=3, edgecolor='black', alpha=alpha)


# ─── App ──────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="ShotTrace", layout="wide")
st.title("ShotTrace")
st.caption("Basketball shot mechanics — pose-only make/miss analysis")

df, keypoints, template = load_shots()
model = load_model()

# Sidebar: shot picker + display options
with st.sidebar:
    st.header("Pick a shot")
    label_filter = st.radio("Filter", ["All", "Makes only", "Misses only"], horizontal=True)
    if label_filter == "Makes only":
        candidates = df[df['label_int'] == 1]
    elif label_filter == "Misses only":
        candidates = df[df['label_int'] == 0]
    else:
        candidates = df

    options = [f"{r['filename'].replace('_shot01_', '_').replace('.npy', '')} "
               f"({'MAKE' if r['label_int']==1 else 'MISS'})"
               for _, r in candidates.iterrows()]
    chosen = st.selectbox("Shot", options)
    # Find the original row index from the chosen string
    chosen_filename = candidates.iloc[options.index(chosen)]['filename']
    idx = int(df.index[df['filename'] == chosen_filename][0])

    st.markdown("---")
    show_template = st.checkbox("Overlay made-shot template", value=True)

shot_kps = keypoints[idx]
shot_row = df.iloc[idx]
true_label_text = "MAKE" if shot_row['label_int'] == 1 else "MISS"
proba           = predict_make_probability(model, shot_kps, int(shot_row['release_frame_est']))
predicted_text  = "MAKE" if proba > 0.5 else "MISS"
distance        = float(np.linalg.norm(shot_kps[:, :, :2].flatten() - template.flatten()))

# Top metrics row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Ground truth", true_label_text)
m2.metric("Model prediction", predicted_text, delta=f"{proba:.0%} make probability")
m3.metric("Distance to template", f"{distance:.1f}")
m4.metric("Shot", f"{idx + 1} / {len(df)}")
correct = (predicted_text == true_label_text)
if correct:
    st.success("Model agrees with the ground-truth label.")
else:
    st.warning("Model disagrees with the ground-truth label — interesting case to inspect.")

# Pose viewer + features
left, right = st.columns([3, 2])

with left:
    st.subheader("Pose")
    frame = st.slider("Frame (0 = 5 frames before motion peak; release lands around frame 5)",
                      0, 29, RELEASE_TARGET_FRAME)

    fig, ax = plt.subplots(figsize=(6, 7))

    if show_template:
        # Template doesn't have a visibility channel — every joint is "valid" for plotting
        plot_skeleton(ax, template[frame], np.ones(9, dtype=bool),
                      color='lightgray', alpha=0.6, linewidth=1.8)

    valid = shot_kps[frame, :, 2] > 0.3
    plot_skeleton(ax, shot_kps[frame, :, :2], valid,
                  color='steelblue', alpha=1.0, linewidth=2.5)

    ax.set_xlim(-3, 3)
    ax.set_ylim(-4, 3)
    ax.set_aspect('equal')
    ax.set_xlabel('x  (shoulder widths from midline)')
    ax.set_ylabel('y  (up is positive)')
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Frame {frame} / 29  —  blue = this shot, gray = made-shot template")
    st.pyplot(fig)

with right:
    st.subheader("Engineered features")
    feat_data = pd.DataFrame({
        'feature': FEATURE_COLS,
        'value':   [round(float(shot_row[c]), 3) for c in FEATURE_COLS],
    })
    st.dataframe(feat_data, use_container_width=True, hide_index=True)

    st.subheader("Wrist trajectory")
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    valid_seq = shot_kps[:, RWRI, 2] > 0.3
    ax2.plot(shot_kps[valid_seq, RWRI, 0], -shot_kps[valid_seq, RWRI, 1],
             color='steelblue', linewidth=2, marker='o', markersize=3)
    ax2.scatter(shot_kps[frame, RWRI, 0], -shot_kps[frame, RWRI, 1],
                color='red', s=80, zorder=3, label=f'frame {frame}')
    ax2.axhline(0, color='gray', linewidth=0.5)
    ax2.axvline(0, color='gray', linewidth=0.5)
    ax2.set_xlabel('x  (shoulder widths)')
    ax2.set_ylabel('y  (up is positive)')
    ax2.set_title('Right wrist over the 30-frame window')
    ax2.legend()
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
