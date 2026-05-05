# Runs MediaPipe Pose on each frame of every trimmed shot and saves keypoints.
# Output: (30, 9, 3) .npy per shot — 30 frames × 9 upper-body keypoints × [x, y, visibility].
# Coordinates are normalized to shoulder-width units, centered on the shoulder midpoint.

import cv2
import numpy as np
import os
import argparse
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# MediaPipe returns 33 landmarks; we keep upper body only. Knees and ankles
# are dropped because baggy clothing makes them unreliable, and lower-body
# motion isn't predictive of shot accuracy anyway.
KEYPOINT_INDICES = [0, 11, 12, 13, 14, 15, 16, 23, 24]
KEYPOINT_NAMES = [
    'nose',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
]
LEFT_SHOULDER, RIGHT_SHOULDER = 1, 2  # indices within the 9 we keep

MODEL_PATH = 'pose_landmarker_full.task'


def setup_pose_detector(model_path):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.IMAGE,
        num_poses=1,
        # Low confidence thresholds — the shooter is small in the frame and the
        # default 0.5 misses too many frames in poorly-lit gym shots.
        min_pose_detection_confidence=0.1,
        min_pose_presence_confidence=0.1,
        min_tracking_confidence=0.1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def extract_frame_keypoints(detector, frame_bgr):
    # MediaPipe wants RGB; OpenCV gives BGR.
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_image)

    if not result.pose_landmarks:
        return np.zeros((len(KEYPOINT_INDICES), 3), dtype=np.float32)

    landmarks = result.pose_landmarks[0]
    return np.array(
        [[landmarks[i].x, landmarks[i].y, landmarks[i].visibility] for i in KEYPOINT_INDICES],
        dtype=np.float32,
    )


def normalize_keypoints(keypoints):
    # Center on shoulder midpoint, scale by shoulder width — removes camera-distance
    # and body-size differences so shots are comparable.
    #
    # Use a single per-shot reference (median over high-visibility frames) instead of
    # normalizing each frame independently. Per-frame normalization explodes whenever
    # MediaPipe places the shoulders near-coincident on a single frame: dividing by a
    # tiny width produces coordinates in the hundreds, which then dominate any .max()-
    # style feature downstream.
    visible = (keypoints[:, LEFT_SHOULDER,  2] > 0.5) & \
              (keypoints[:, RIGHT_SHOULDER, 2] > 0.5)
    if visible.sum() < 3:
        return keypoints  # not enough good frames; eda.py's visibility filter will drop this shot

    good = keypoints[visible]
    ls = np.median(good[:, LEFT_SHOULDER,  :2], axis=0)
    rs = np.median(good[:, RIGHT_SHOULDER, :2], axis=0)
    center = (ls + rs) / 2.0
    width  = np.linalg.norm(rs - ls)

    if width < 0.02:  # shoulders coincident across the whole shot — unfixable, leave raw
        return keypoints

    normalized = keypoints.copy()
    normalized[:, :, :2] = (keypoints[:, :, :2] - center) / width
    return normalized


def process_shot(detector, npy_path, out_path):
    frames = np.load(npy_path)  # (30, H, W, 3)
    keypoints = np.stack(
        [extract_frame_keypoints(detector, frames[i]) for i in range(frames.shape[0])],
        axis=0,
    )
    keypoints = normalize_keypoints(keypoints)
    np.save(out_path, keypoints)
    return keypoints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trimmed_dir', required=True)
    parser.add_argument('--out_dir', required=True)
    parser.add_argument('--model', default=MODEL_PATH)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.model):
        print(f"Model file not found: {args.model}")
        print("Download pose_landmarker_full.task from:")
        print("https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker")
        return

    detector = setup_pose_detector(args.model)

    npy_files = [f for f in sorted(Path(args.trimmed_dir).glob('*.npy'))
                 if 'manifest' not in f.name]
    print(f"Found {len(npy_files)} shot files\n")

    for i, npy_path in enumerate(npy_files):
        out_path = os.path.join(args.out_dir, npy_path.name)
        if os.path.exists(out_path):
            print(f"[{i+1}/{len(npy_files)}] {npy_path.name} (already done)")
            continue

        print(f"[{i+1}/{len(npy_files)}] {npy_path.name}")
        kps = process_shot(detector, str(npy_path), out_path)
        detected = (kps[:, :, 2].mean(axis=1) > 0.3).sum()
        print(f"  pose detected in {detected}/30 frames")

    detector.close()
    print(f"\nDone. Keypoints saved to {args.out_dir}/")


if __name__ == '__main__':
    main()
