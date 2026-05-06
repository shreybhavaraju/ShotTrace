# Consolidates the cross-validated AUC + accuracy from every experiment we ran into a
# single comparison figure and CSV. Hard-codes the numbers — re-running every model
# would take ~10 min and produce the same values (same seeds, same data).

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

FIGS = Path('figs')
FIGS.mkdir(exist_ok=True)

# Headline results — only the runs the current pipeline (baseline.py + model.py)
# actually produces, so anything plotted here is reproducible. Earlier exploration
# runs (6-feature baselines, intermediate CNN versions) were dropped to keep the
# comparison focused on what's worth presenting.
RESULTS = [
    ('Logistic regression', 'baseline',  222, 0.436, 0.039, 0.446, 0.026),
    ('Random forest',       'baseline',  222, 0.459, 0.052, 0.496, 0.047),
    ('Gradient boosting',   'baseline',  222, 0.520, 0.055, 0.518, 0.047),
    ('1D CNN (aligned)',    'sequence',  222, 0.596, 0.092, 0.608, 0.058),
]

df = pd.DataFrame(RESULTS, columns=['model', 'family', 'n_shots',
                                     'auc_mean', 'auc_std', 'acc_mean', 'acc_std'])
df.to_csv(FIGS / 'all_models_results.csv', index=False)

family_colors = {
    'baseline': '#7fb069',
    'sequence': '#c45a5a',
}
df = df.sort_values('auc_mean').reset_index(drop=True)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, metric, title, label in [
    (axes[0], 'auc',  'Cross-validated AUC',      'AUC'),
    (axes[1], 'acc',  'Cross-validated accuracy', 'accuracy'),
]:
    means = df[f'{metric}_mean']
    stds  = df[f'{metric}_std']
    colors = [family_colors[f] for f in df['family']]

    ax.barh(df['model'], means, xerr=stds, color=colors, edgecolor='black',
            capsize=4, error_kw={'linewidth': 1.2})
    ax.axvline(0.5, color='gray', linestyle='--', label='chance', alpha=0.7)
    ax.set_xlim(0.4, 0.65)
    ax.set_xlabel(f'mean {label} (±1 std across 5 folds)')
    ax.set_title(title)
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(m + s + 0.005, i, f'{m:.3f}', va='center', fontsize=9)

# One legend for the figure
handles = [plt.Rectangle((0, 0), 1, 1, color=c, edgecolor='black') for c in family_colors.values()]
fig.legend(handles, family_colors.keys(),
           loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02))
plt.suptitle('Model comparison — 5-fold cross-validated results across all experiments',
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(FIGS / 'all_models_comparison.png', dpi=120, bbox_inches='tight')
plt.close()


# Markdown table for the writeup / slides
md_lines = [
    '| Model | Dataset | AUC (CV) | Accuracy (CV) |',
    '|---|---|---|---|',
]
for _, r in df.sort_values('auc_mean', ascending=False).iterrows():
    md_lines.append(
        f"| {r['model']} | {r['n_shots']} shots | {r['auc_mean']:.3f} ± {r['auc_std']:.3f} | "
        f"{r['acc_mean']:.3f} ± {r['acc_std']:.3f} |"
    )

(FIGS / 'all_models_results.md').write_text('\n'.join(md_lines) + '\n')

print(f"saved comparison figure to {FIGS}/all_models_comparison.png")
print(f"saved results CSV to       {FIGS}/all_models_results.csv")
print(f"saved markdown table to    {FIGS}/all_models_results.md")
print()
print("results, sorted by AUC:")
print(df.sort_values('auc_mean', ascending=False)[
    ['model', 'n_shots', 'auc_mean', 'auc_std']
].to_string(index=False))
