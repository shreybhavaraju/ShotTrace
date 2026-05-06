# Floor-check baseline: simple sklearn models on the engineered features.
# If random forest can't beat ~55% here, the features lack signal and the CNN won't save it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
)
from sklearn.calibration import calibration_curve

SEED = 42
FEATURES = Path('trimmed/features.csv')
FIGS = Path('figs')
FIGS.mkdir(exist_ok=True)

FEATURE_COLS = ['max_elbow_extension', 'release_frame_est',
                'peak_wrist_height',  'peak_wrist_frame',
                'drive_amplitude',    'follow_through_x']

df = pd.read_csv(FEATURES)
X = df[FEATURE_COLS].values
y = df['label_int'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED,
)
print(f"train: {len(X_train)} shots ({y_train.mean():.1%} makes)")
print(f"test:  {len(X_test)} shots ({y_test.mean():.1%} makes)\n")


def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    print(f"=== {name} ===")
    print(f"  accuracy: {accuracy_score(y_test, pred):.3f}")
    print(f"  AUC:      {roc_auc_score(y_test, proba):.3f}")
    print(f"  confusion matrix [[TN FP] [FN TP]]:")
    print(f"    {confusion_matrix(y_test, pred).tolist()}")
    print(classification_report(y_test, pred, target_names=['miss', 'make']))
    return {
        'pred':  pred,
        'proba': proba,
        'acc':   accuracy_score(y_test, pred),
        'auc':   roc_auc_score(y_test, proba),
    }


# Logistic regression gets a StandardScaler in front because the engineered features
# live on wildly different scales (degrees, shoulder-widths, frame index 0-29) — without
# scaling, the largest-scale feature would dominate the loss surface. Random forest
# is scale-invariant so it doesn't need this.
lr = Pipeline([
    ('scale', StandardScaler()),
    ('lr',    LogisticRegression(max_iter=1000, random_state=SEED)),
])
rf = RandomForestClassifier(n_estimators=200, random_state=SEED)

lr_res = evaluate('logistic regression', lr)
rf_res = evaluate('random forest',       rf)

# 5-fold cross-validation — single 80/20 split has ~±0.075 SE on AUC at this sample size,
# so any one split is mostly noise. CV gives a stable estimate to actually decide on.
print("=== 5-fold cross-validation (full dataset) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, model in [('logistic regression', lr), ('random forest', rf)]:
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"  {name}:")
    print(f"    accuracy: {acc.mean():.3f} ± {acc.std():.3f}   per-fold: {[f'{a:.2f}' for a in acc]}")
    print(f"    AUC:      {auc.mean():.3f} ± {auc.std():.3f}   per-fold: {[f'{a:.2f}' for a in auc]}")
print()

# Use whichever model has better AUC for the visualizations
better_name, better = ('random forest', rf_res) if rf_res['auc'] >= lr_res['auc'] \
                      else ('logistic regression', lr_res)


# Confusion matrix figure
cm = confusion_matrix(y_test, better['pred'])
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['miss', 'make'])
ax.set_yticklabels(['miss', 'make'])
ax.set_xlabel('predicted')
ax.set_ylabel('true')
ax.set_title(f'Confusion matrix — {better_name}')
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(FIGS / 'baseline_confusion.png', dpi=120)
plt.close()


# Feature importance — random forest only. Logistic regression coefficients on
# StandardScaler-transformed features are technically comparable, but RF's importance
# is more straightforward to explain and more robust to feature correlation.
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 4))
importances.plot.barh(color='steelblue', edgecolor='black')
plt.xlabel('importance')
plt.title('Random forest feature importance')
plt.tight_layout()
plt.savefig(FIGS / 'baseline_feature_importance.png', dpi=120)
plt.close()
print("feature importances (random forest, descending):")
print(importances.sort_values(ascending=False).round(3))


# Calibration: are the model's confidence numbers actually meaningful?
# Quantile binning (equal samples per bin) is stabler than uniform on a ~56-shot test set
# where uniform bins often end up empty.
prob_true, prob_pred = calibration_curve(y_test, better['proba'], n_bins=5, strategy='quantile')

plt.figure(figsize=(6, 5))
plt.plot([0, 1], [0, 1], '--', color='gray', label='perfect calibration')
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=better_name)
plt.xlabel('predicted make probability')
plt.ylabel('actual make rate')
plt.xlim(0, 1); plt.ylim(0, 1)
plt.title('Calibration')
plt.legend()
plt.tight_layout()
plt.savefig(FIGS / 'baseline_calibration.png', dpi=120)
plt.close()

print(f"\nfigures saved to {FIGS}/")
