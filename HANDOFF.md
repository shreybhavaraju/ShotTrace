# ShotTrace — Handoff Document

This is a complete state-of-the-project doc written for handoff to a new assistant. It assumes zero prior context.

---

## 1. What this project is

**ShotTrace** is a UC Berkeley **Data Foundations** class project (freshman level) that predicts whether a basketball shot will go in *based purely on body mechanics* — pose data only, **no ball tracking**. The constraint is the whole point: a model that learns from body motion could in principle be a coaching tool ("your form is breaking down"), while ball tracking just tells you what already happened.

The full pipeline is built end-to-end. Best model achieved is a small 1D CNN at **AUC 0.596 ± 0.092** (5-fold CV), which beats the engineered-feature baselines (best is gradient boosting at AUC 0.520).

The most important *finding* of the project: **pose alone, at 278/222 shots from a single shooter, runs into a data ceiling**. Three model classes (LR, RF, GB) on 14 engineered features all converge to AUC ≈ 0.46–0.52. The CNN sees signal the engineered features miss (+0.05 AUC), but it's a small win. The honest conclusion is that you need either (a) more shooters, (b) more shots, or (c) ball trajectory data to push performance further.

---

## 2. Constraints to honor

- **Class project scope.** No over-engineering. No fancy architectures. Comments should explain *why* not *what*. No banner dividers in code. Code should look like a competent freshman wrote it.
- **No ball tracking.** This is a deliberate framing decision. Adding ball would change the project's identity and erase its main finding. Ball tracking is mentioned as a *future-work pivot* in the slides, not a thing to actually implement.
- **Workflow: one file per turn, user commits manually.** Never run `git commit` for the user. After each meaningful change, stop and explicitly flag "this is a commit point." User wants the git history to look like a real student building incrementally.
- **The user is a UC Berkeley freshman in a Data Foundations class.** Frame technical explanations accordingly.

---

## 3. Tech stack

Python with: `numpy`, `pandas`, `matplotlib`, `opencv-python`, `mediapipe`, `scikit-learn`, `pytorch`, `streamlit`. All standard. No GPU required.

---

## 4. File structure

```
ShotTrace/
├── label_shots.py             # 1. Interactive labeling (M=make, X=miss)
├── shot_pipeline.py           # 2. Trim raw .MOV to 30-frame window around release
├── extract_keypoints.py       # 3. MediaPipe Pose → (30, 9, 3) keypoint arrays
├── eda.py                     # 4. EDA + visibility filter → manifest_clean.csv
├── features.py                # 5. Per-shot engineered features (14) → features.csv
├── baseline.py                # 6. LR + RF + GB on features (with 5-fold CV)
├── model.py                   # 7. 1D CNN on keypoint sequences (with 5-fold CV)
├── analysis.py                # 8. "Good shot" template + form drift plot
├── results_summary.py         # 9. Consolidated comparison figure
├── app.py                     # 10. Streamlit dashboard
├── pose_landmarker_full.task  # MediaPipe model file
├── model.pt                   # Saved CNN weights from best fold
├── raw_videos/                # 328 .MOV files, labels in filename (_make/_miss)
├── trimmed/
│   ├── manifest.csv           # 328 shots, full metadata
│   ├── manifest_clean.csv     # 298 shots passing visibility filter
│   ├── features.csv           # 222 shots after outlier filter (the modeling set)
│   ├── good_shot_template.npy # (30, 9, 2) — average of made-shot keypoints
│   └── *.npy                  # 328 trimmed video clips, shape (30, 1920, 1080, 3)
├── keypoints/
│   └── *.npy                  # 328 keypoint files, shape (30, 9, 3)
└── figs/                      # All output figures (5 EDA + baseline + CNN + comparison + drift)
```

---

## 5. Data shapes and conventions

| Stage | Object | Shape | Meaning |
|---|---|---|---|
| Trimmed video | `trimmed/*.npy` | `(30, 1920, 1080, 3)` | 30 frames × H × W × BGR |
| Keypoints | `keypoints/*.npy` | `(30, 9, 3)` | 30 frames × 9 joints × `[x, y, visibility]` |
| Features | `features.csv` | 222 rows × 14 cols + `filename` + `label_int` | Per-shot engineered features |
| CNN input | (computed in `model.py`) | `(N, 30, 36)` | 30 frames × (9 joints × 2 coords + 18 velocities) |
| Template | `good_shot_template.npy` | `(30, 9, 2)` | Mean over made shots, no visibility |

**The 9 keypoints** (indices 0–8): `nose, L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist, L_hip, R_hip`. Right side is the shooting arm. Knees and ankles are intentionally dropped (baggy clothing made them unreliable, lower-body isn't the dominant signal).

**Coordinates are normalized:** `(x, y)` are in shoulder-width units, centered on the shoulder midpoint. `y` increases *down* (image convention) — flipping to `-y` makes plots read like a person.

**Frame conventions:**
- Frame 0 = 5 frames before motion-peak release
- Frame 5 = motion-peak release (originally assumed to be the actual release; **EDA showed it's not** — it's mid-extension)
- Frame 29 = 25 frames after motion-peak release
- After temporal re-alignment in `model.py`, peak elbow extension lands at **frame 15**

**Joint indices in code:** `NOSE=0, LSHO=1, RSHO=2, LELB=3, RELB=4, LWRI=5, RWRI=6, LHIP=7, RHIP=8`.

---

## 6. The pipeline, file by file

### `label_shots.py`
Plays each unlabeled `.MOV` in a loop, you press **M** (make) or **X** (miss), it renames the file in place to embed the label. Saves progress to JSON for resume. Already done — 162 makes, 166 misses.

### `shot_pipeline.py`
Detects the release frame by finding the **biggest motion spike in the first half of the video** (left 45% of the frame, where the shooter stands). Refines ±8 frames using a tighter upper-arm crop. Trims to **5 frames before + 25 after = 30 total**. Outputs `trimmed/*.npy` + `manifest.csv`.

### `extract_keypoints.py`
Runs MediaPipe Pose (full model, low confidence thresholds at 0.1) on every frame. Keeps 9 upper-body keypoints. **Critical fix from the original version:** normalization uses a *single per-shot reference* (median shoulders across high-visibility frames) rather than per-frame normalization, which used to explode coordinates whenever MediaPipe placed shoulders near-coincident on a single frame. Width threshold is 0.005 (lowered from 0.02 because shooters far from camera have legitimately small shoulder widths).

### `eda.py`
Five sections:
1. **Visibility filter.** Mean visibility on R_shoulder/R_elbow/R_wrist, threshold 0.5 → drops 30 shots, leaves 298.
2. **Class balance check.** Pre-filter 49.4% makes; post-filter 49.3% — filter didn't bias the dataset.
3. **Wrist + elbow trajectory overlay** (makes vs misses).
4. **Elbow angle at frame 5.** *This is the key finding* — angles are 122° ± 35° for both makes and misses. Frame 5 isn't the actual release; the motion-peak detector finds mid-extension. This drove the design of `features.py` to use release-invariant features (max-over-sequence rather than measured at a fixed frame).
5. **Make rate by session.** Tried to infer sessions from gaps in IMG numbering; failed (no gaps > 20). Currently shows one giant "session." Could be fixed by reading `.MOV` mtime, but it's not on the critical path.

Output: `trimmed/manifest_clean.csv` (298 shots, with engineered columns) + 5 PNGs in `figs/`.

### `features.py`
Computes 14 release-invariant features per shot. Each is computed across all 30 frames using `.max() / .argmax() / .min()` — never measured at a fixed frame.

The 14 features:
- `max_elbow_extension`, `release_frame_est` (argmax of elbow angle), `peak_wrist_height`, `peak_wrist_frame` (argmin of wrist y), `drive_amplitude`, `follow_through_x` — the original 6
- `left_max_elbow_ext`, `guide_hand_asymmetry`, `max_wrist_velocity`, `max_arm_whip_speed`, `release_to_peak_lag`, `wrist_above_head`, `shoulder_tilt_at_peak`, `post_release_drop` — the 8 added later

**Outlier filter:** drops shots where `drive_amplitude > 20` or `|follow_through_x| > 20`. These are normalization blowups — physically impossible magnitudes, and they were poisoning every model. Drops ~16 shots; final dataset is 222 shots.

Output: `trimmed/features.csv` (222 rows × 14 features + filename + label_int).

### `baseline.py`
Loads `features.csv`, runs a stratified 80/20 split + 5-fold cross-validation. Three models:
1. **Logistic regression** (with `StandardScaler` in a pipeline because features have very different scales)
2. **Random forest** (200 trees)
3. **Gradient boosting** (200 estimators, max_depth=3)

Saves `figs/baseline_confusion.png`, `figs/baseline_feature_importance.png`, `figs/baseline_calibration.png`. Random `random_state=42` everywhere a split happens, for reproducibility.

### `model.py`
1D CNN on keypoint sequences. Reads `trimmed/features.csv` to know which shots to load, then loads the keypoint files for those.

**Preprocessing pipeline** (in `load_data()`):
1. Load `(N, 30, 9, 2)` positions (drop visibility channel)
2. Reshape to `(N, 30, 18)`
3. Compute frame-to-frame velocities via `np.diff` with prepend → another `(N, 30, 18)`
4. Concatenate channels → `(N, 30, 36)`
5. **Re-align temporally** so peak elbow extension lands at frame 15 (target). This puts every shot on the same kinematic clock — different shots had release at different positions in the original window, which forced the CNN to learn alignment + classification simultaneously. Out-of-range frames get padded with the nearest edge frame (zero velocity at the boundaries).

**Architecture:**
```
Conv1d(36 → 32, kernel=5, padding=2) → ReLU → Dropout(0.3)
Conv1d(32 → 32, kernel=5, padding=2) → ReLU → Dropout(0.3)
AdaptiveAvgPool1d(1) → Flatten → Linear(32 → 1)
```
BCEWithLogitsLoss + Adam (lr=1e-3) + Gaussian noise augmentation on training inputs (std=0.05).

**Training:** 5-fold stratified CV, 50 epochs per fold. Saves the highest-AUC fold's weights to `model.pt` and the training curves of that fold to `figs/cnn_training_curves.png`.

### `analysis.py`
Computes the **"good shot" template** as the mean of all made-shot keypoint sequences → `(30, 9, 2)`. Computes Euclidean distance from each shot to the template. Plots two panels: distance histogram makes-vs-misses + rolling mean of distance over chronological shot index (form drift).

Output: `trimmed/good_shot_template.npy`, `figs/template_distance_drift.png`.

### `results_summary.py`
Hard-codes the cross-validated results from the four "headline" models (LR, RF, GB on engineered features + the CNN on aligned sequences) and produces a single comparison figure with error bars. Outputs: `figs/all_models_comparison.png`, `figs/all_models_results.csv`, `figs/all_models_results.md`.

### `app.py`
Streamlit dashboard. Reads `features.csv`, all keypoints, the template, and `model.pt`. Sidebar lets the user pick any of 222 shots (filterable to makes-only / misses-only). Main panel shows ground-truth label, model prediction with probability, distance to template, a frame slider with a stick-figure pose plot (with optional template overlay), the engineered features as a table, and the right wrist trajectory plot. The **green/yellow banner** that flags whether the model agreed with the ground truth doubles as a built-in error-analysis tool — clicking through the yellow ones is the easiest way to inspect what the CNN gets wrong.

Run with: `streamlit run app.py`.

---

## 7. Every experiment we tried, in order

This is the chronological story of the modeling work. Each row is a real experiment with cross-validated numbers.

| # | Experiment | Dataset | CV AUC | Notes |
|---|---|---|---|---|
| 1 | LR, 6 features | 278 | 0.518 ± 0.043 | Original baseline |
| 2 | RF, 6 features | 278 | 0.457 ± 0.073 | Worse than LR — odd |
| 3 | CNN v1 (positions only) | 278 | 0.527 ± 0.031 | First neural attempt |
| 4 | CNN v2 (velocities + dropout + noise=0.05) | 278 | 0.527 ± 0.038 | Closed train-test gap from 0.19 → 0.07 but didn't improve AUC. **Confirmed engineered features weren't masking signal.** |
| 5 | LR, 14 features | 222 | 0.436 ± 0.039 | Adding features hurt LR (noise dilution) |
| 6 | RF, 14 features | 222 | 0.459 ± 0.052 | Same as 6-feature version |
| 7 | GB, 14 features | 222 | 0.520 ± 0.055 | Best engineered-features result |
| 8 | CNN v2 on 222-shot subset (no realignment) | 222 | 0.576 ± 0.035 | **First meaningful improvement.** The 56 shots dropped were also confusing the CNN. |
| 9 | CNN v2 + temporal realignment | 222 | 0.596 ± 0.092 | **Best result.** Mean up but variance tripled (high per-fold variance). |
| 10 | LR/RF/GB with template-distance features (leakage-free) | 222 | All ≈ 0.43–0.50 | **Negative result.** Distance to make/miss templates didn't add signal beyond the 14 engineered features. Important to note: I built the template *inside each fold* from train-only makes; if you do it naively from all data it leaks test labels and inflates AUC. |

**Best model:** CNN v2 with realignment, AUC 0.596 ± 0.092 (5-fold CV). Stored in `model.pt`, training curves in `figs/cnn_training_curves.png`.

---

## 8. Important gotchas / non-obvious decisions

- **Frame 5 ≠ release.** The motion-peak detector finds the moment of max arm motion (mid-extension), not full extension. Don't measure features at frame 5 — use `.max()/.argmax()` across the sequence.
- **The visibility filter is not a substitute for the magnitude filter.** Some shots pass the visibility threshold (mean visibility on R-arm ≥ 0.5) but still have individual frames where MediaPipe failed (vis=0, coords=(0,0)). Those poison `argmin(wrist_y)` because zero is more negative than any real frame's wrist position. **Always mask out vis=0 frames before reduce-over-time operations.** Done in `features.py:shot_features()`.
- **Per-shot normalization, not per-frame.** Originally `extract_keypoints.py` normalized each frame independently; on jittery frames where MediaPipe placed shoulders near-coincident, dividing by a tiny shoulder-width exploded the coordinates. Fixed by computing one reference per shot from the median of high-visibility frames.
- **The outlier filter in `features.py` drops 16 shots.** These are normalization blowups (`drive_amplitude > 20` or `|follow_through_x| > 20`). They're physically impossible magnitudes and were dragging models around. Filter is at the feature level, not the keypoint level — keypoint files for those shots still exist on disk; they're just not in `features.csv`.
- **Random_state=42 everywhere.** Both `baseline.py` and `model.py` use the same random seed for the same `StratifiedKFold`, so their CV folds line up. This is what makes the AUC numbers directly comparable.
- **Ensembling and template-distance features were both tried and didn't help meaningfully.** The CNN's advantage over engineered features is *temporal*, not summary-statistics. We have evidence for this from experiment 10 — distance-to-template (a spatial summary) added zero AUC.
- **The 5 EDA figures (`figs/01_*` through `figs/05_*`) are referenced by the writeup story.** Don't delete or rename them.

---

## 9. What's currently committed / state of the codebase

All files listed in section 4 are in working order. Last meaningful changes:
- `analysis.py` no longer dumps `distances.csv` (was unused)
- `baseline.py` reverted to remove the leakage-free template-distance experiment (didn't help)
- `pose_landmarker_lite.task` deleted (unused — only `pose_landmarker_full.task` is loaded)
- `app.py` is brand-new and tested-imports-work

If you run the full pipeline cold:
```bash
# (assuming raw_videos/ already labeled)
python shot_pipeline.py --batch_dir ./raw_videos/   # ~30 min
python extract_keypoints.py --trimmed_dir ./trimmed/ --out_dir ./keypoints/   # ~20 min
python eda.py                                       # quick
python features.py                                  # quick
python baseline.py                                  # ~30 sec
python model.py                                     # ~3 min (5-fold CV × 50 epochs)
python analysis.py                                  # quick
python results_summary.py                           # quick
streamlit run app.py                                # opens browser
```

In day-to-day work you only re-run the parts that changed (e.g., `features.py` + `baseline.py` if features changed; `model.py` if keypoints or model code changed).

---

## 10. What's left to do

In rough priority order:

1. **Polish `app.py`.** It works but hasn't been tested with the user actually using it. Possible polish: improve the pose-plot styling, add a "show worst predictions" filter (sort by `|proba - label|` descending), maybe add the wrist trajectory of the template overlaid on the shot's trajectory.
2. **Write the slide deck / final writeup.** The narrative is:
   - Constraint: pose-only, no ball
   - Pipeline: video → trimmed → keypoints → features → models → app (compression from 250 MB to 3 KB per shot)
   - EDA finding that drove the plan: elbow at frame 5 doesn't separate makes/misses (frame 5 isn't really the release)
   - Modeling: 4 models (LR, RF, GB, CNN), 5-fold CV everywhere, best is CNN at 0.596 AUC
   - Honest finding: pose-only hits a data ceiling at 222 shots from one shooter
   - Future work: multi-shooter dataset (addresses the data ceiling), ball trajectory (addresses the physics gap), saliency maps (turns predictions into coaching feedback)
3. **Optional: manual error analysis.** Use the `app.py` yellow-banner cases — pull up shots the model got wrong, write up patterns. Great for the "what I learned by inspecting my errors" slide.
4. **Optional: CNN + GB ensemble.** Quick experiment that could push AUC from 0.596 toward 0.61. Implement by training both inside the same CV loop and averaging probabilities.
5. **Optional: fix session detection in `eda.py`.** Switch from IMG-number gaps to file mtime gaps. Would let `analysis.py` show real form drift between filming sessions instead of one giant series.

User explicitly said *not* to add ball tracking — it stays as a "future work" slide topic only.

---

## 11. Code style notes

- Top-of-file: 2–3 line comment, no docstring walls
- No banner-style section dividers (`# ─── STEP N ───`)
- Comments only when the *why* is non-obvious; never restate what the code does
- Variable names natural, no `*Service` or `*Manager` indirection
- Don't add error handling for cases that can't happen in this pipeline
- Single-shooter, single-machine — no need for defensive validation at internal boundaries

The user must be able to walk through any function and explain it in plain English.

---

## 12. Workflow

- **One file per turn.** After making a meaningful change, stop and report. Don't pre-write the next file.
- **User commits manually.** Never run `git add` or `git commit`. Explicitly flag commit-worthy stopping points and suggest a commit message.
- **No `--no-verify`, no force pushes.** Standard git hygiene.
- **The user's environment:** macOS, zsh, Python in a conda base env, MediaPipe and Streamlit already installed, PyTorch CPU-only is fine.

---

## 13. Quick-start command sequence to verify state

After picking this up in Cursor, run these to confirm everything still works:

```bash
# Verify the CNN loads and makes a prediction
python -c "
import torch, numpy as np, pandas as pd
from model import ShotCNN, realign_to_release
m = ShotCNN(); m.load_state_dict(torch.load('model.pt', weights_only=True)); m.eval()
df = pd.read_csv('trimmed/features.csv')
kps = np.load(f\"keypoints/{df.iloc[0]['filename']}\")
positions = kps[:, :, :2].reshape(1, 30, 18)
velocities = np.diff(positions, axis=1, prepend=positions[:, :1, :])
X = realign_to_release(np.concatenate([positions, velocities], axis=2),
                       [int(df.iloc[0]['release_frame_est'])])
print('prob:', float(torch.sigmoid(m(torch.tensor(X, dtype=torch.float32))).item()))
"

# Run the dashboard
streamlit run app.py
```

If both work, the project is in the same state as when the previous session ended.

---

## 14. The headline numbers for the slides

- Dataset: 328 raw shots → 298 after pose-detection visibility filter → 222 after feature-magnitude outlier filter
- Class balance: 49.1% makes (109/113)
- Best baseline (engineered features): Gradient Boosting, AUC 0.520 ± 0.055, accuracy 0.518
- Best model (sequence): 1D CNN with temporal alignment, AUC 0.596 ± 0.092, accuracy 0.608
- Train/test gap on the CNN: 0.10 (mild overfitting; below 0.15 threshold)
- The CNN's advantage over GB on the same 222-shot dataset: +0.076 AUC — small but real, and the +0.076 is *because of the temporal sequence*, not summary statistics (verified by the failed template-distance experiment).
