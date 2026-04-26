import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.size'] = 11

out_dir = 'data/output/metrics'
os.makedirs(out_dir, exist_ok=True)

# Data
classes = ['Unlikely (Minority)', 'Partial (Majority)', 'Highly Likely']
before_smote_f1 = [0.471, 0.859, 0.809]
after_smote_f1 = [0.784, 0.842, 0.831]

x = np.arange(len(classes))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5.5))

bars1 = ax.bar(x - width/2, before_smote_f1, width, label='Before SMOTE (Baseline Fine-Tuned)',
               color='#94A3B8', edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x + width/2, after_smote_f1, width, label='After SMOTE (Balanced Embeddings)',
               color='#10B981', edgecolor='white', linewidth=1.5)

for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height()*100:.1f}%', ha='center', va='bottom', fontsize=10, color='#64748B')
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
            f'{bar.get_height()*100:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#047857')

ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
ax.set_title('Impact of SMOTE on Per-Class F1-Scores', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(classes, fontsize=11)
ax.set_ylim(0.0, 1.0)
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.grid(True, axis='y', alpha=0.3, linestyle='-')

# Add annotation for the massive jump
ax.annotate('+31.3% Improvement',
            xy=(0 + width/2, 0.784), xytext=(0.5, 0.90),
            arrowprops=dict(arrowstyle='->', color='#047857', lw=2),
            fontsize=10, color='#047857', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#D1FAE5', edgecolor='#10B981', alpha=0.9))

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'smote_comparison_bars.png'), dpi=200, bbox_inches='tight')
print("Saved SMOTE graph.")
