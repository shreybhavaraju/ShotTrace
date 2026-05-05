# Plays each video in a window and labels it via keyboard.
# M=make, X=miss, R=replay, S=skip, Q=quit. Renames files in place and saves
# progress to JSON so labeling can be resumed across sessions.

import cv2
import os
import json
import argparse
from pathlib import Path

PROGRESS_FILE = 'labeling_progress.json'

KEYS = {'m': 'make', 'x': 'miss', 'r': 'replay', 's': 'skip', 'q': 'quit'}


def load_progress(folder):
    path = os.path.join(folder, PROGRESS_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'labeled': {}, 'skipped': []}


def save_progress(folder, progress):
    with open(os.path.join(folder, PROGRESS_FILE), 'w') as f:
        json.dump(progress, f, indent=2)


def get_unlabeled_videos(folder, progress):
    all_videos = sorted(Path(folder).glob('*.mov')) + \
                 sorted(Path(folder).glob('*.MOV')) + \
                 sorted(Path(folder).glob('*.mp4'))
    done = set(progress['labeled'].keys()) | set(progress['skipped'])
    # Files already renamed with _make/_miss are also "done", even if a
    # progress.json got deleted — rely on the filename as ground truth.
    return [v for v in all_videos
            if v.name not in done
            and 'make' not in v.name.lower()
            and 'miss' not in v.name.lower()]


def play_and_label(video_path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = int(1000 / fps)
    window = 'Shot Labeler  |  M=make  X=miss  R=replay  S=skip  Q=quit'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 400, 700)

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break  # video ended → outer loop restarts playback

            h, w = frame.shape[:2]
            display = cv2.resize(frame, (int(w * 700 / h), 700))
            cv2.imshow(window, display)

            key = cv2.waitKey(delay) & 0xFF
            if key < 128:
                action = KEYS.get(chr(key).lower())
                if action == 'replay':
                    break
                if action:
                    cap.release()
                    cv2.destroyAllWindows()
                    return action


def rename_with_label(video_path, label):
    new_path = video_path.parent / f"{video_path.stem}_{label}{video_path.suffix}"
    video_path.rename(new_path)
    return new_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    args = parser.parse_args()

    folder = args.folder
    progress = load_progress(folder)
    videos = get_unlabeled_videos(folder, progress)

    total_done = len(progress['labeled'])
    total_all = total_done + len(videos)

    print(f"\n{len(videos)} unlabeled videos  ({total_done} already labeled)")
    print("Controls: M=make  X=miss  R=replay  S=skip  Q=quit\n")

    for i, video_path in enumerate(videos):
        print(f"[{total_done + i + 1}/{total_all}]  {video_path.name}  ({len(videos) - i} left)")

        action = play_and_label(video_path)

        if action in ('make', 'miss'):
            new_path = rename_with_label(video_path, action)
            progress['labeled'][video_path.name] = action
            save_progress(folder, progress)
            print(f"  → {action.upper()}  ({new_path.name})")

        elif action == 'skip':
            progress['skipped'].append(video_path.name)
            save_progress(folder, progress)
            print(f"  → SKIPPED")

        elif action == 'quit':
            save_progress(folder, progress)
            print(f"\nProgress saved. {total_done + i} labeled so far.")
            break

    makes = sum(1 for v in progress['labeled'].values() if v == 'make')
    misses = sum(1 for v in progress['labeled'].values() if v == 'miss')
    print(f"\nDone! Makes: {makes}  Misses: {misses}  Total: {makes + misses}")


if __name__ == '__main__':
    main()
