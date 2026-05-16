#!/usr/bin/env python3
from pathlib import Path
import csv
import matplotlib.pyplot as plt
import numpy as np
import shutil

ROOT = Path('/workspaces/Baseline')
RESULTS = ROOT / 'results'
FIG_DIR = RESULTS / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Read comparison csv
comp_csv = RESULTS / 'threshold_comparison.csv'
systems = []
metrics = []
with comp_csv.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        systems.append(row['system'])
        metrics.append({k: float(v) for k, v in row.items() if k!='system'})

# AUC and pAUC bar chart
labels = [s for s in systems]
x = np.arange(len(labels))
width = 0.35
auc = [m['AUC_total'] for m in metrics]
pauc = [m['pAUC'] for m in metrics]
fig, ax = plt.subplots(figsize=(8,4.2))
ax.bar(x - width/2, auc, width, label='AUC_total', color='tab:blue')
ax.bar(x + width/2, pauc, width, label='pAUC', color='tab:orange')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylim(0,1)
ax.set_ylabel('Score')
ax.set_title('AUC and partial AUC comparison')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / 'comparison_auc_pauc.png', dpi=300)
fig.savefig(FIG_DIR / 'comparison_auc_pauc.pdf')
plt.close(fig)

# F1 source/target grouped bar
f1s = [m['F1_source'] for m in metrics]
f1t = [m['F1_target'] for m in metrics]
fig, ax = plt.subplots(figsize=(8,4.2))
ax.bar(x - width/2, f1s, width, label='F1_source', color='tab:green')
ax.bar(x + width/2, f1t, width, label='F1_target', color='tab:red')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')
ax.set_ylim(0,1)
ax.set_ylabel('F1')
ax.set_title('F1 (source vs target)')
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / 'comparison_f1_source_target.png', dpi=300)
fig.savefig(FIG_DIR / 'comparison_f1_source_target.pdf')
plt.close(fig)

# Methodology diagram (simple boxes)
fig, ax = plt.subplots(figsize=(8,3))
ax.axis('off')
ax.text(0.05,0.6,'Audio Files', bbox=dict(boxstyle='round', facecolor='lightgrey'), fontsize=12)
ax.text(0.35,0.6,'Feature Extraction\n(mels)', bbox=dict(boxstyle='round', facecolor='lightgrey'), fontsize=12)
ax.text(0.65,0.6,'Autoencoder\n(Train on normals)', bbox=dict(boxstyle='round', facecolor='lightgrey'), fontsize=12)
ax.text(0.85,0.6,'Anomaly Scores\nThresholding', bbox=dict(boxstyle='round', facecolor='lightgrey'), fontsize=12)
# arrows
ax.annotate('', xy=(0.2,0.6), xytext=(0.14,0.6), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.47,0.6), xytext=(0.39,0.6), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.78,0.6), xytext=(0.69,0.6), arrowprops=dict(arrowstyle='->', lw=2))
ax.set_title('Methodology Pipeline')
fig.tight_layout()
fig.savefig(FIG_DIR / 'methodology_pipeline.png', dpi=300)
fig.savefig(FIG_DIR / 'methodology_pipeline.pdf')
plt.close(fig)

# Architecture figure (AE schematic)
fig, ax = plt.subplots(figsize=(6,3))
ax.axis('off')
ax.text(0.1,0.5,'Input (mels)', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.text(0.4,0.5,'Encoder\n(bottleneck)', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.text(0.7,0.5,'Decoder\n(reconstruction)', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.annotate('', xy=(0.32,0.5), xytext=(0.18,0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.62,0.5), xytext=(0.48,0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.set_title('Autoencoder Architecture (schematic)')
fig.tight_layout()
fig.savefig(FIG_DIR / 'autoencoder_architecture.png', dpi=300)
fig.savefig(FIG_DIR / 'autoencoder_architecture.pdf')
plt.close(fig)

# Threshold calibration workflow
fig, ax = plt.subplots(figsize=(8,3))
ax.axis('off')
ax.text(0.05,0.5,'Validation normal scores', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.text(0.35,0.5,'Compute percentiles\nand mean+ksd', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.text(0.65,0.5,'Evaluate on validation\nselect best by F1', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.text(0.92,0.5,'Apply per-domain\nthresholds to test', bbox=dict(boxstyle='round', facecolor='lightgrey'))
ax.annotate('', xy=(0.25,0.5), xytext=(0.16,0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.55,0.5), xytext=(0.38,0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.annotate('', xy=(0.84,0.5), xytext=(0.68,0.5), arrowprops=dict(arrowstyle='->', lw=2))
ax.set_title('Threshold Calibration Workflow')
fig.tight_layout()
fig.savefig(FIG_DIR / 'threshold_calibration_workflow.png', dpi=300)
fig.savefig(FIG_DIR / 'threshold_calibration_workflow.pdf')
plt.close(fig)

# Copy existing ROC/PR and domain figures into results/figures for inclusion
src_figs = [
    ROOT / 'results' / 'eval_data' / 'temporal_dynamic_threshold_MSE' / 'temporal_dynamic_threshold_roc.png',
    ROOT / 'results' / 'eval_data' / 'temporal_dynamic_threshold_MSE' / 'temporal_dynamic_threshold_score_hist.png',
    ROOT / 'results' / 'eval_data' / 'temporal_domain_threshold' / 'domain_separate_roc.png',
    ROOT / 'results' / 'eval_data' / 'temporal_domain_threshold' / 'fpr_calibration_roc.png',
]
for s in src_figs:
    if s.exists():
        shutil.copy(s, FIG_DIR / s.name)

# Generate markdown files
md_base = RESULTS / 'final_thesis_analysis.md'
with md_base.open('w') as f:
    f.write('# Final Thesis Analysis\n\n')
    f.write('See figures in `results/figures/`.\n\n')
    f.write('## Key results\n')
    f.write('- Baseline AE (MSE): AUC_total = 0.5493, pAUC = 0.5358\n')
    f.write('- Baseline + Global Normalization (best overall): AUC_total = 0.5766, pAUC = 0.5153\n')
    f.write('- GlobalNorm + Temporal (best target/pAUC): AUC_target = 0.5376, pAUC = 0.5537\n')
    f.write('- GlobalNorm + Temporal (domain-calibrated): AUC_total = 0.5143, F1_source = 0.2997, F1_target = 0.3392\n')
    f.write('\n')
    f.write('![AUC comparison](figures/comparison_auc_pauc.png)\n')
    f.write('![F1 comparison](figures/comparison_f1_source_target.png)\n')
    f.write('![ROC dynamic](figures/temporal_dynamic_threshold_roc.png)\n')

for name, content in [('final_conclusion.md','## Conclusion\n\nThe results show that global normalization improves overall detection performance...'),
                      ('final_discussion.md','## Discussion\n\nThe experiments indicate domain shift challenges...'),
                      ('future_work.md','## Future Work\n\nExplore lightweight domain adaptation techniques...'),
                      ('limitations.md','## Limitations\n\nLimited to existing datasets and fixed models...')]:
    p = RESULTS / name
    with p.open('w') as f:
        f.write(content + '\n')

print('Generated thesis materials in', RESULTS)
print('Figures in', FIG_DIR)
