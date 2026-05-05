# Floor-check baseline: simple sklearn models on the engineered features.
# If random forest can't beat ~55% here, the features lack signal and the CNN won't save it.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
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


lr_model = LogisticRegression(max_iter=1000, random_state=SEED)
rf_model = RandomForestClassifier(n_estimators=200, random_state=SEED)

lr = evaluate('logistic regression', lr_model)
rf = evaluate('random forest',       rf_model)

# Use the better-AUC model for the confusion / calibration plots
better_name, better = ('random forest', rf) if rf['auc'] >= lr['auc'] else ('logistic regression', lr)


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


# Feature importance — RF only. LR coefficients on unscaled features aren't directly comparable
# across features that live on wildly different scales (degrees vs shoulder-widths vs frame index).
importances = pd.Series(rf_model.feature_importances_, index=FEATURE_COLS).sort_values()
plt.figure(figsize=(8, 4))
importances.plot.barh(color='steelblue', edgecolor='black')
plt.xlabel('importance')
plt.title('Random forest feature importance')
plt.tight_layout()
plt.savefig(FIGS / 'baseline_feature_importance.png', dpi=120)
plt.close()
print("feature importances (random forest, descending):")
print(importances.sort_values(ascending=False).round(3))


# Calibration plot — quantile strategy (equal-sample bins) because the test set is only ~60 shots
# and uniform bins tend to leave bins empty.
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
