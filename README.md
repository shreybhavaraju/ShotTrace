# ShotTrace

**Basketball shot mechanics — pose-only make/miss prediction using MediaPipe + 1D CNN**

ShotTrace is a personal data science project that asks a simple question: can you predict whether a basketball shot goes in from nothing but body pose? No ball tracking, no launch angle sensors — just skeletal keypoints extracted from phone video.

The answer turns out to be: kind of. The 1D CNN achieves **AUC 0.660** on 222 hand-labeled shots, well above the engineered-feature baselines (best: gradient boosting at 0.520). Details below.

---

## Results

| Model | AUC (5-fold CV) | Accuracy (5-fold CV) |
|---|---|---|
| 1D CNN (aligned) | **0.660 ± 0.066** | **0.617 ± 0.049** |
| Gradient boosting | 0.520 ± 0.055 | 0.518 ± 0.047 |
| Random forest | 0.459 ± 0.052 | 0.496 ± 0.047 |
| Logistic regression | 0.436 ± 0.039 | 0.446 ± 0.026 |

The CNN's edge over the tabular baselines is meaningful: engineered features summarize each shot with ~14 numbers, losing all timing and coordination information. The CNN sees all 30 frames at once and can pick up on things like joint velocity profiles and arm-whip timing that don't reduce to a single statistic.

---

## Pipeline Overview

```
raw phone videos (.MOV)
        ↓
  shot_pipeline.py       — trim to 30-frame window around release
        ↓
  label_shots.py         — keyboard labeling (M=make, X=miss)
        ↓
  extract_keypoints.py   — MediaPipe Pose → (30, 9, 3) .npy per shot
        ↓
  eda.py                 — visibility filtering, manifest_clean.csv
        ↓
  features.py            — 14 engineered features → features.csv
        ↓
  baseline.py            — LR / RF / GB on engineered features
  model.py               — 1D CNN on full keypoint sequences
        ↓
  analysis.py            — form-drift analysis over chronological shot order
  app.py                 — Streamlit interactive viewer
```

---

## File Descriptions

| File | What it does |
|---|---|
| `shot_pipeline.py` | Detects the release frame in each labeled `.MOV` via motion-spike analysis, trims a 30-frame window, and writes `(30, H, W, 3)` `.npy` files + `manifest.csv`. |
| `label_shots.py` | Interactive keyboard labeler. Plays each video and records make/miss via keypresses. Saves progress to JSON so sessions can be resumed. |
| `extract_keypoints.py` | Runs MediaPipe Pose on each trimmed shot. Keeps 9 upper-body joints (nose, shoulders, elbows, wrists, hips), normalizes coordinates to shoulder-width units, and saves `(30, 9, 3)` `.npy` files. |
| `eda.py` | Filters shots with poor arm visibility, plots rolling make rate and elbow angle distributions, writes `manifest_clean.csv`. |
| `features.py` | Computes 14 per-shot features (max elbow extension, wrist velocity, drive amplitude, guide-hand asymmetry, etc.) and writes `features.csv`. Drops shots with broken normalization. |
| `baseline.py` | Trains logistic regression, random forest, and gradient boosting on the 14 engineered features. Reports 5-fold CV AUC/accuracy and feature importances. |
| `model.py` | Trains the 1D CNN. Includes temporal re-alignment (shifts each shot so peak elbow extension lands at frame 15), mirror-flip augmentation, Gaussian jitter, and 5-fold CV. Saves the best fold's weights to `model.pt`. |
| `analysis.py` | Builds a "good shot" template by averaging made-shot keypoint sequences. Plots per-shot distance to the template over chronological order to check for form drift with fatigue. |
| `app.py` | Streamlit dashboard. Browse any shot, scrub through frames, view the pose skeleton, elbow angle curve, wrist trajectory, engineered features, and the CNN's prediction. Includes a "worst predictions" filter for error analysis. |
| `results_summary.py` | Generates the comparison bar chart and markdown table from hard-coded CV results. |

---

## Setup

```bash
pip install mediapipe torch torchvision scikit-learn pandas numpy matplotlib streamlit
```

Download the MediaPipe pose model:
```
pose_landmarker_full.task
```
Available at https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker

---

## Running the Pipeline

**Step 1 — Label videos** (if starting from raw `.MOV` files):
```bash
python label_shots.py --folder raw_videos/
```

**Step 2 — Trim to 30-frame windows:**
```bash
python shot_pipeline.py --batch_dir raw_videos/ --out_dir trimmed/
```

**Step 3 — Extract pose keypoints:**
```bash
python extract_keypoints.py --trimmed_dir trimmed/ --out_dir keypoints/
```

**Step 4 — EDA and filtering:**
```bash
python eda.py
```

**Step 5 — Compute engineered features:**
```bash
python features.py
```

**Step 6 — Train models:**
```bash
python baseline.py   # tabular baselines
python model.py      # 1D CNN (saves model.pt)
```

**Step 7 — Analysis and visualization:**
```bash
python analysis.py
python results_summary.py
```

**Step 8 — Launch the dashboard:**
```bash
streamlit run app.py
```

---

## Data Notes

- **222 shots** after visibility filtering (shots where MediaPipe couldn't reliably detect the shooting arm were dropped).
- Shots are from a single shooter (right-handed), filmed from a fixed camera position.
- Coordinates are normalized to shoulder-width units centered on the shoulder midpoint — removes camera-distance and body-size variation across shots.
- The `keypoints/` and `trimmed/` directories (raw frames, `.npy` files) are gitignored. Only the processed `features.csv` and `model.pt` are tracked.

---

## Model Details

**Architecture:**
```
Conv1d(36→32, k=5) → ReLU → Dropout(0.3)
Conv1d(32→32, k=5) → ReLU → Dropout(0.3)
AdaptiveAvgPool1d(1) → Flatten → Linear(32→1)
```

Input: `(batch, 30, 36)` — 30 frames × (18 position + 18 velocity channels). Transposed to `(batch, 36, 30)` for `Conv1d`.

**Training:**
- 50 epochs, Adam (lr=1e-3), BCEWithLogitsLoss
- Mirror-flip augmentation doubles the training set each fold
- Gaussian jitter (σ=0.05) on keypoints during training
- Temporal alignment: each shot shifted so peak elbow extension lands at frame 15

---

## Limitations

- 222 shots is a small dataset for a sequence model — the ±0.066 std on AUC reflects real fold-to-fold variance, not just noise.
- Single shooter, single camera angle. Generalization to other shooters or setups is untested.
- MediaPipe occasionally loses the wrist on fast-motion frames; failed detections are linearly interpolated before model input.
- The model predicts shot outcome from body mechanics only — it has no information about the ball's trajectory after release.
