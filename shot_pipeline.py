# Trims each labeled .MOV to a 30-frame window around the release frame.
# Release is detected as the largest motion spike in the shooter region (left
# side of the frame). Output: (30, H, W, 3) .npy per shot + a manifest.csv.

import cv2
import numpy as np
import os
import csv
import argparse
from pathlib import Path

FRAMES_BEFORE = 5
FRAMES_AFTER = 25
WINDOW_SIZE = FRAMES_BEFORE + FRAMES_AFTER

# Shooter is on the left ~45% of the frame. Restricting motion analysis here
# avoids false peaks from the ball flying or other people moving on the right.
SHOOTER_X = 0.45
SHOOTER_Y = 0.85


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps


def motion_signal(frames):
    # Mean abs pixel diff between consecutive frames in the shooter region.
    # The shooting motion produces a clear spike when the arm accelerates up.
    H, W = frames[0].shape[:2]
    sx, sy = int(W * SHOOTER_X), int(H * SHOOTER_Y)
    motion = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        motion.append(float(gray[:sy, :sx].mean()))
    return np.array(motion)


def find_release_frame(frames):
    # Pick the highest motion peak in the first half of the video. The shot
    # is always the biggest, earliest event — anything later is rebound/walk-back.
    motion = motion_signal(frames)
    rough = int(np.argmax(motion[:len(motion) // 2])) + 1

    # Refine ±8 frames using a tighter upper-arm crop. The wrist/arm spike is
    # sharper than the whole-shooter-region signal so the peak is more precise.
    H, W = frames[0].shape[:2]
    arm_x, arm_y = int(W * 0.30), int(H * 0.55)
    start = max(1, rough - 8)
    end = min(len(frames) - 1, rough + 8)

    best_frame, best_score = rough, -1.0
    for i in range(start, end + 1):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        score = float(gray[:arm_y, :arm_x].mean())
        if score > best_score:
            best_frame, best_score = i, score
    return best_frame


def trim_shot(frames, release):
    # If release sits near the start/end of the video, pad by repeating the
    # nearest valid frame so the output is always exactly WINDOW_SIZE frames.
    start = release - FRAMES_BEFORE
    end = release + FRAMES_AFTER
    return np.stack(
        [frames[max(0, min(len(frames) - 1, i))] for i in range(start, end)],
        axis=0,
    )


def process_video(video_path, label, out_dir, manifest_rows):
    stem = Path(video_path).stem
    print(f"\nProcessing: {stem}  (label={label})")

    frames, fps = load_frames(str(video_path))
    print(f"  {len(frames)} frames @ {fps:.1f} fps  ({len(frames)/fps:.1f}s)")

    if len(frames) < WINDOW_SIZE + 5:
        print(f"  Too short, skipping")
        return

    release = find_release_frame(frames)
    trimmed = trim_shot(frames, release)

    out_name = f"{stem}_shot01_{label}.npy"
    np.save(os.path.join(out_dir, out_name), trimmed)
    print(f"  release={release}  →  saved {out_name}  shape={trimmed.shape}")

    manifest_rows.append({
        'filename': out_name,
        'source_video': Path(video_path).name,
        'shot_index_in_video': 1,
        'label': label,
        'label_int': 1 if label == 'make' else 0,
        'release_frame': release,
        'fps': round(fps, 2),
        'total_source_frames': len(frames),
    })


def main():
    parser = argparse.ArgumentParser(description='Basketball shot trimming pipeline')
    parser.add_argument('--video')
    parser.add_argument('--label', choices=['make', 'miss'])
    parser.add_argument('--batch_dir', help='Directory of videos (filenames must contain make/miss)')
    parser.add_argument('--out_dir', default='./trimmed')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'manifest.csv')

    # Carry forward existing rows so reruns append rather than overwrite.
    existing = []
    if os.path.exists(manifest_path):
        with open(manifest_path) as f:
            existing = list(csv.DictReader(f))
        print(f"Loaded {len(existing)} existing manifest rows")

    new_rows = []
    if args.video and args.label:
        process_video(args.video, args.label, args.out_dir, new_rows)
    elif args.batch_dir:
        videos = sorted(list(Path(args.batch_dir).glob('*.mov')) +
                        list(Path(args.batch_dir).glob('*.MOV')) +
                        list(Path(args.batch_dir).glob('*.mp4')))
        print(f"Found {len(videos)} videos in {args.batch_dir}")
        for vf in videos:
            name = vf.name.lower()
            if 'make' in name:
                label = 'make'
            elif 'miss' in name:
                label = 'miss'
            else:
                print(f"  Skipping {vf.name} — no make/miss in filename")
                continue
            process_video(str(vf), label, args.out_dir, new_rows)
    else:
        parser.print_help()
        return

    all_rows = existing + new_rows
    if all_rows:
        with open(manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nManifest: {len(all_rows)} total shots → {manifest_path}")

    makes = sum(1 for r in all_rows if r['label'] == 'make')
    misses = sum(1 for r in all_rows if r['label'] == 'miss')
    print(f"Dataset: {makes} makes, {misses} misses, {makes + misses} total")


if __name__ == '__main__':
    main()
