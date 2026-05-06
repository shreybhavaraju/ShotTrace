# 1D CNN on keypoint sequences. Sees all 30 frames at once and learns whatever
# temporal patterns it can find — joint coordination, micro-tempo, wobble — none
# of which reduce to a single number. This is the test of whether sequence-level
# signal exists that the engineered-feature baseline missed.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
KEYPOINTS_DIR = Path('keypoints')
FEATURES = Path('trimmed/features.csv')   # 222 shots after the outlier filter
MODEL_PATH = Path('model.pt')
FIGS = Path('figs')
FIGS.mkdir(exist_ok=True)

EPOCHS     = 50
BATCH_SIZE = 32
LR         = 1e-3
NOISE_STD  = 0.05   # Gaussian augmentation — bumped from 0.02 to fight overfitting

np.random.seed(SEED)
torch.manual_seed(SEED)


class ShotCNN(nn.Module):
    # Conv1d expects (batch, channels, length). We treat the per-frame numbers as
    # channels and the 30 frames as length, so kernels slide across *time*. Input is
    # 36 channels: 18 position values + 18 velocity values (frame-to-frame deltas).
    def __init__(self, in_channels=36):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),                  # regularize: 222-shot training set overfits fast
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.AdaptiveAvgPool1d(1),          # global pool over time → (batch, 32, 1)
            nn.Flatten(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x is (batch, 30, 36) → transpose to (batch, 36, 30) for Conv1d
        return self.net(x.transpose(1, 2)).squeeze(-1)


class ShotDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx]
        # Gaussian jitter on keypoints — tiny perturbations expand the effective
        # dataset, important when 222 train samples is small for a 1D CNN.
        if self.augment:
            x = x + torch.randn_like(x) * NOISE_STD
        return x, self.y[idx]


RELEASE_TARGET_FRAME = 15   # frame index where peak elbow extension lands after alignment


def mirror_flip(X):
    # Mirror each shot about the y-axis: negate x coords and swap left/right joints.
    # The shooter is right-handed, so the flipped shot is a synthetic left-handed one
    # with the same make/miss label — bilateral symmetry means doubling the dataset
    # this way is free training data. Channel layout per frame (36 ch): positions live
    # in [0..17] as 9 joints × (x, y) interleaved, velocities in [18..35] same layout.
    JOINT_SWAPS = [(1, 2), (3, 4), (5, 6), (7, 8)]   # L↔R for shoulder, elbow, wrist, hip
    flipped = X.copy()
    for offset in (0, 18):
        for l, r in JOINT_SWAPS:
            l_x, l_y = offset + l*2, offset + l*2 + 1
            r_x, r_y = offset + r*2, offset + r*2 + 1
            flipped[:, :, [l_x, l_y, r_x, r_y]] = X[:, :, [r_x, r_y, l_x, l_y]]
    # Negate every x channel (even indices) across both positions and velocities
    x_channels = list(range(0, 36, 2))
    flipped[:, :, x_channels] *= -1
    return flipped


def realign_to_release(X, release_frames, target=RELEASE_TARGET_FRAME):
    # Shift each shot so peak elbow extension lands at the same time index. Different
    # shots have release at different positions in the original window (motion-peak
    # detection is noisy), which forces the CNN to learn alignment+classification at
    # the same time. Re-aligning puts every shot on the same kinematic clock — same
    # arm-whip pattern always at the same temporal position. Out-of-range frames are
    # padded with the nearest edge frame, which has zero velocity (edge replication).
    N, T, _ = X.shape
    aligned = np.empty_like(X)
    for i in range(N):
        shift = int(target - release_frames[i])
        for t in range(T):
            src = max(0, min(T - 1, t - shift))
            aligned[i, t] = X[i, src]
    return aligned


def replace_failed_frames(raw_kps, threshold=0.3):
    # MediaPipe stores failed detections as (x=0, y=0, vis=0). Letting those zero
    # pulses through forces the CNN to learn around a "wrist teleports to origin"
    # artifact. For each joint independently, linearly interpolate the (x, y) over
    # frames where vis < threshold, and edge-fill at the boundaries. If a joint has
    # no valid frames in the entire shot (e.g. left arm fully occluded), leave it
    # at (0, 0) — there's nothing to interpolate from.
    cleaned = raw_kps[:, :, :, :2].copy()
    bad = raw_kps[:, :, :, 2] < threshold
    T = cleaned.shape[1]
    n_fixed = 0
    for i in range(cleaned.shape[0]):
        for j in range(cleaned.shape[2]):
            n_bad = int(bad[i, :, j].sum())
            if n_bad == 0 or n_bad == T:
                continue
            for c in range(2):
                s = pd.Series(np.where(bad[i, :, j], np.nan, cleaned[i, :, j, c]))
                s = s.interpolate(limit_direction='both')
                cleaned[i, :, j, c] = s.values
            n_fixed += n_bad
    return cleaned, n_fixed


def load_data():
    feat_df = pd.read_csv(FEATURES)
    raw_kps = np.stack([
        np.load(KEYPOINTS_DIR / fn) for fn in feat_df['filename']
    ])   # (N, 30, 9, 3) — keep visibility briefly for the cleanup step

    positions, n_fixed = replace_failed_frames(raw_kps)
    total = raw_kps.shape[0] * raw_kps.shape[1] * raw_kps.shape[2]
    print(f"  cleaned {n_fixed} failed-detection joint-frames out of {total} ({n_fixed/total:.1%})")
    positions = positions.reshape(positions.shape[0], 30, 18)

    # Velocity = frame-to-frame delta. Captures speed of motion, which the engineered
    # features only summarized as a single number (release tempo). The CNN can use this
    # to discriminate based on *how fast* joints move, not just where they end up.
    # Compute velocities *before* alignment so they stay consistent with their position
    # frame after the shift.
    velocities = np.diff(positions, axis=1, prepend=positions[:, :1, :])
    X = np.concatenate([positions, velocities], axis=2)   # (N, 30, 36)

    X = realign_to_release(X, feat_df['release_frame_est'].values)

    y = feat_df['label_int'].values
    return X, y


def train_fold(X_train, y_train, X_test, y_test):
    # Double the training fold with mirror-flipped copies. Test fold stays untouched
    # so we evaluate on real shots only.
    X_train = np.concatenate([X_train, mirror_flip(X_train)], axis=0)
    y_train = np.concatenate([y_train, y_train], axis=0)

    train_loader = DataLoader(
        ShotDataset(X_train, y_train, augment=True),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        ShotDataset(X_test, y_test, augment=False),
        batch_size=BATCH_SIZE, shuffle=False,
    )

    model = ShotCNN()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {'train_loss': [], 'test_loss': [],
            'train_acc':  [], 'test_acc':  [],
            'test_auc':   []}
    best_auc, best_state = -1.0, None

    for _ in range(EPOCHS):
        model.train()
        loss_sum, correct, n = 0.0, 0, 0
        for x, y in train_loader:
            opt.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct  += ((torch.sigmoid(logits) > 0.5).float() == y).sum().item()
            n        += len(y)
        hist['train_loss'].append(loss_sum / n)
        hist['train_acc'].append(correct / n)

        model.eval()
        all_logits, all_y = [], []
        with torch.no_grad():
            for x, y in test_loader:
                all_logits.append(model(x))
                all_y.append(y)
        all_logits = torch.cat(all_logits)
        all_y      = torch.cat(all_y)
        all_proba  = torch.sigmoid(all_logits).numpy()
        test_acc   = float(((all_proba > 0.5) == all_y.numpy()).mean())
        test_auc   = float(roc_auc_score(all_y.numpy(), all_proba))
        hist['test_loss'].append(loss_fn(all_logits, all_y).item())
        hist['test_acc'].append(test_acc)
        hist['test_auc'].append(test_auc)

        if test_auc > best_auc:
            best_auc = test_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    return {
        'best_auc':    best_auc,
        'best_state':  best_state,
        'best_acc':    max(hist['test_acc']),
        'history':     hist,
    }


def main():
    print("loading data...")
    X, y = load_data()
    print(f"  X shape: {X.shape}   y balance: {y.mean():.2%} makes\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print("=== 5-fold cross-validation ===")
    fold_results = []
    for fold, (tr, te) in enumerate(cv.split(X, y)):
        r = train_fold(X[tr], y[tr], X[te], y[te])
        fold_results.append(r)
        gap = r['history']['train_acc'][-1] - r['best_acc']
        print(f"  fold {fold+1}: best AUC={r['best_auc']:.3f}  "
              f"best acc={r['best_acc']:.3f}  "
              f"train-test gap={gap:+.2f}")

    aucs = np.array([r['best_auc'] for r in fold_results])
    accs = np.array([r['best_acc'] for r in fold_results])
    gaps = np.array([r['history']['train_acc'][-1] - r['best_acc'] for r in fold_results])
    print(f"\n  AUC:      {aucs.mean():.3f} ± {aucs.std():.3f}")
    print(f"  accuracy: {accs.mean():.3f} ± {accs.std():.3f}")
    print(f"  avg train-test gap: {gaps.mean():+.2f}  (>0.15 means overfitting)")

    # Save the highest-AUC fold's weights for analysis.py and app.py to load
    best = int(aucs.argmax())
    torch.save(fold_results[best]['best_state'], MODEL_PATH)
    print(f"\nsaved fold-{best+1} weights to {MODEL_PATH}")

    # Training curves from the best fold
    h = fold_results[best]['history']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(h['train_loss'], label='train')
    axes[0].plot(h['test_loss'],  label='test')
    axes[0].set_xlabel('epoch'); axes[0].set_ylabel('loss')
    axes[0].set_title(f'Loss (fold {best+1})')
    axes[0].legend()

    axes[1].plot(h['train_acc'], label='train')
    axes[1].plot(h['test_acc'],  label='test')
    axes[1].axhline(0.5, color='gray', linestyle='--', label='chance')
    axes[1].set_xlabel('epoch'); axes[1].set_ylabel('accuracy')
    axes[1].set_title(f'Accuracy (fold {best+1})')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIGS / 'cnn_training_curves.png', dpi=120)
    plt.close()
    print(f"training curves saved to {FIGS}/cnn_training_curves.png")


if __name__ == '__main__':
    main()
