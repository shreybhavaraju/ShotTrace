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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, classification_report,
)

SEED = 42
FEATURES = Path('trimmed/features.csv')
FIGS = Path('figs')
FIGS.mkdir(exist_ok=True)

FEATURE_COLS = ['max_elbow_extension', 'release_frame_est',
                'peak_wrist_height',  'peak_wrist_frame',
                'drive_amplitude',    'follow_through_x',
                'left_max_elbow_ext', 'guide_hand_asymmetry',
                'max_wrist_velocity', 'max_arm_whip_speed',
                'release_to_peak_lag', 'wrist_above_head',
                'shoulder_tilt_at_peak', 'post_release_drop']

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
# Gradient boosting often beats RF on small, structured tabular data — it handles
# noisy/non-informative features better via additive shallow trees instead of bagging.
gb = GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=SEED)

lr_res = evaluate('logistic regression', lr)
rf_res = evaluate('random forest',       rf)
gb_res = evaluate('gradient boosting',   gb)

# 5-fold cross-validation — single 80/20 split has ~±0.075 SE on AUC at this sample size,
# so any one split is mostly noise. CV gives a stable estimate to actually decide on.
print("=== 5-fold cross-validation (full dataset) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
for name, model in [('logistic regression', lr), ('random forest', rf), ('gradient boosting', gb)]:
    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    auc = cross_val_score(model, X, y, cv=cv, scoring='roc_auc')
    print(f"  {name}:")
    print(f"    accuracy: {acc.mean():.3f} ± {acc.std():.3f}   per-fold: {[f'{a:.2f}' for a in acc]}")
    print(f"    AUC:      {auc.mean():.3f} ± {auc.std():.3f}   per-fold: {[f'{a:.2f}' for a in auc]}")
print()

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

print(f"\nfigure saved to {FIGS}/baseline_feature_importance.png")
