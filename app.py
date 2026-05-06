# Streamlit dashboard. Static viewer over the 222 cleaned shots — pick any shot,
# scrub through frames to see the pose, the engineered features, the elbow-angle
# curve, and the CNN's prediction. The "worst predictions" filter sorts by
# |proba − label|, so the app doubles as a built-in error-analysis tool.

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
MODEL_PATH    = Path('model.pt')

NOSE = 0
LSHO, RSHO = 1, 2
LELB, RELB = 3, 4
LWRI, RWRI = 5, 6
LHIP, RHIP = 7, 8

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
    return df, keypoints


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


@st.cache_resource
def compute_all_predictions(_model, _keypoints, _release_frames):
    # Precompute every shot's predicted probability so the "worst predictions"
    # filter can sort by model error without re-running inference on every reload.
    probas = np.zeros(len(_release_frames))
    for i, rf in enumerate(_release_frames):
        probas[i] = predict_make_probability(_model, _keypoints[i], int(rf))
    return probas


def compute_elbow_angles(shot_kps, threshold=0.3):
    # Right elbow angle per frame, NaN where any of the three joints fails. Same
    # formula as features.py but inlined here so the dashboard doesn't import it.
    angles = np.full(30, np.nan)
    for f in range(30):
        if (shot_kps[f, [RSHO, RELB, RWRI], 2] >= threshold).all():
            sho = shot_kps[f, RSHO, :2]
            elb = shot_kps[f, RELB, :2]
            wri = shot_kps[f, RWRI, :2]
            v1, v2 = sho - elb, wri - elb
            cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            angles[f] = np.degrees(np.arccos(np.clip(cos, -1, 1)))
    return angles


def view_window(shot_kps, padding=0.4):
    # Each shot is normalized using its own shoulder width, so different shots
    # land on very different visible scales. Auto-zoom to the bounding box of
    # this shot's valid joints across all 30 frames so the pose is always
    # legible regardless of the per-shot scale.
    valid = shot_kps[:, :, 2] > 0.3
    if not valid.any():
        return -1, 1, -1, 1
    coords = shot_kps[:, :, :2][valid]
    x_min, x_max = coords[:, 0].min(), coords[:, 0].max()
    y_min, y_max = coords[:, 1].min(), coords[:, 1].max()
    return x_min - padding, x_max + padding, -y_max - padding, -y_min + padding


def plot_skeleton(ax, kps_xy, valid, color, alpha=1.0, linewidth=2.5):
    for a, b in SKELETON:
        if valid[a] and valid[b]:
            ax.plot([kps_xy[a, 0], kps_xy[b, 0]],
                    [-kps_xy[a, 1], -kps_xy[b, 1]],
                    color=color, linewidth=linewidth, alpha=alpha)
    for j in range(9):
        if valid[j]:
            ax.scatter(kps_xy[j, 0], -kps_xy[j, 1],
                       color=color, s=45, zorder=3, edgecolor='black', alpha=alpha)


st.set_page_config(page_title="ShotTrace", layout="wide")
st.title("ShotTrace")
st.caption("Basketball shot mechanics — pose-only make/miss analysis")

df, keypoints = load_shots()
model = load_model()
probas = compute_all_predictions(model, keypoints, df['release_frame_est'].values)
df['proba'] = probas
df['error'] = np.abs(probas - df['label_int'])

with st.sidebar:
    st.header("Pick a shot")
    label_filter = st.radio(
        "Filter",
        ["All", "Makes only", "Misses only", "Worst predictions"],
        help="‘Worst predictions’ sorts shots by how confidently wrong the model was — the model's most interesting errors live at the top.",
    )
    if label_filter == "Makes only":
        candidates = df[df['label_int'] == 1].copy()
    elif label_filter == "Misses only":
        candidates = df[df['label_int'] == 0].copy()
    elif label_filter == "Worst predictions":
        candidates = df.sort_values('error', ascending=False).head(50).copy()
    else:
        candidates = df.copy()

    options = []
    for _, r in candidates.iterrows():
        name = r['filename'].replace('_shot01_', '_').replace('.npy', '')
        label_str = 'MAKE' if r['label_int'] == 1 else 'MISS'
        if label_filter == "Worst predictions":
            options.append(f"{name}  ({label_str}, proba={r['proba']:.2f})")
        else:
            options.append(f"{name}  ({label_str})")
    chosen = st.selectbox("Shot", options)
    chosen_filename = candidates.iloc[options.index(chosen)]['filename']
    idx = int(df.index[df['filename'] == chosen_filename][0])

shot_kps   = keypoints[idx]
shot_row   = df.iloc[idx]
true_text  = "MAKE" if shot_row['label_int'] == 1 else "MISS"
proba      = float(shot_row['proba'])
pred_text  = "MAKE" if proba > 0.5 else "MISS"

m1, m2, m3, m4 = st.columns(4)
m1.metric("Ground truth", true_text)
m2.metric("Model prediction", pred_text)
m3.metric("Make probability", f"{proba:.1%}")
m4.metric("Shot", f"{idx + 1} / {len(df)}")
if pred_text == true_text:
    st.success("Model agrees with the ground-truth label.")
else:
    st.warning("Model disagrees with the ground-truth label — interesting case to inspect.")

left, right = st.columns([3, 2])

with left:
    st.subheader("Pose")
    frame = st.slider(
        "Frame  (release lands around frame 15 after temporal alignment)",
        0, 29, RELEASE_TARGET_FRAME,
    )

    fig, ax = plt.subplots(figsize=(6, 7))
    valid = shot_kps[frame, :, 2] > 0.3
    plot_skeleton(ax, shot_kps[frame, :, :2], valid,
                  color='steelblue', alpha=1.0, linewidth=2.5)

    x0, x1, y0, y1 = view_window(shot_kps)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect('equal')
    ax.set_xlabel('x  (shoulder widths from midline)')
    ax.set_ylabel('y  (up is positive)')
    ax.grid(True, alpha=0.3)
    n_valid = int(valid.sum())
    ax.set_title(f"Frame {frame} / 29  —  {n_valid}/9 joints visible")
    st.pyplot(fig)

with right:
    st.subheader("Engineered features")
    feat_data = pd.DataFrame({
        'feature': FEATURE_COLS,
        'value':   [round(float(shot_row[c]), 3) for c in FEATURE_COLS],
    })
    st.dataframe(feat_data, use_container_width=True, hide_index=True, height=520)

st.markdown("---")
chart_left, chart_right = st.columns(2)

with chart_left:
    st.subheader("Right elbow angle")
    angles = compute_elbow_angles(shot_kps)
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
    ax2.plot(np.arange(30), angles, color='steelblue', linewidth=2, marker='o', markersize=3)
    ax2.axvline(int(shot_row['release_frame_est']), color='green', linestyle='--',
                alpha=0.7, label='estimated release')
    ax2.axvline(frame, color='red', linewidth=1.5, alpha=0.8, label=f'frame {frame}')
    ax2.axhline(180, color='gray', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('frame')
    ax2.set_ylabel('elbow angle (°)')
    ax2.set_ylim(0, 200)
    ax2.set_title('Right elbow extension over the shot')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

with chart_right:
    st.subheader("Right wrist trajectory")
    fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
    valid_seq = shot_kps[:, RWRI, 2] > 0.3
    ax3.plot(shot_kps[valid_seq, RWRI, 0], -shot_kps[valid_seq, RWRI, 1],
             color='steelblue', linewidth=2, marker='o', markersize=3)
    if valid_seq[frame]:
        ax3.scatter(shot_kps[frame, RWRI, 0], -shot_kps[frame, RWRI, 1],
                    color='red', s=80, zorder=3, label=f'frame {frame}')
    ax3.set_xlabel('x  (shoulder widths)')
    ax3.set_ylabel('y  (up is positive)')
    ax3.set_title('Wrist position over the 30-frame window')
    ax3.legend(fontsize=8)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)
