"""
Generate publication-quality figures for Nature Microbiology paper.
"""
import pickle, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Nature-style aesthetics
rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'legend.fontsize': 7,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

OUT_DIR = "results"
PLOT_DIR = "../paper/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# Load results
with open(f"{OUT_DIR}/benchmark_microbial_final.json") as f:
    results = json.load(f)

bf = results["benchmark_80_20"]["results"]
loco = results["leave_one_class_out"]["results"]

# ============================================================
# FIGURE 1: Benchmark overview
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

# Panel A: Bar chart
names = [r["name"].replace("ESMC-SAE ", "").replace(" binary", "\nbin")
          .replace(" weights", "\nwgt").replace(" binary+weights", "bin+\nwgt")
          .replace(" (", "\n(").replace("3-mer binary", "3-mer\nseq") for r in bf]
top1 = [r["top1"] for r in bf]
top5 = [r["top5"] for r in bf]
xpos = np.arange(len(names))
colors = ["#7f7f7f", "#3b82f6", "#10b981", "#ef4444"]

ax = axes[0]
w = 0.35
b1 = ax.bar(xpos - w/2, top1, w, color=colors, edgecolor='white', linewidth=0.3, alpha=0.9)
b2 = ax.bar(xpos + w/2, [0]*4, w, color=['none']*4)  # placeholder

# Add top-1 labels
for i, (bar, val) in enumerate(zip(b1, top1)):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=6, fontweight='bold')

# Add top-5 as text overlay
for i, (t1, t5) in enumerate(zip(top1, top5)):
    ax.text(xpos[i], t1 - 0.08, f'Top-5: {t5:.3f}', ha='center', va='top',
            fontsize=5.5, color='white', fontweight='bold')

ax.set_xticks(xpos)
ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("EC3 prediction (161 classes, 4,868 enzymes)", fontweight='bold')
ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Leave-one-class-out
ax = axes[1]
loco_names = ["3-mer\nsequence", "ESMC-SAE\nbinary", "ESMC-SAE\nweights"]
loco_vals = [l["ec1_loco"] for l in loco]
loco_colors = ["#7f7f7f", "#3b82f6", "#10b981"]
xpos2 = np.arange(len(loco_names))

bars = ax.bar(xpos2, loco_vals, 0.5, color=loco_colors, edgecolor='white', linewidth=0.3, alpha=0.9)
for bar, val in zip(bars, loco_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
            f'{val/0.1429:.1f}x', ha='center', va='center', fontsize=6,
            color='white', fontweight='bold')

ax.axhline(y=0.1429, color='#d62728', linestyle='--', linewidth=0.8, label='Random (0.143)')
ax.set_xticks(xpos2)
ax.set_xticklabels(loco_names, fontsize=6.5)
ax.set_ylabel("EC1 Recovery Accuracy")
ax.set_ylim(0, 0.65)
ax.set_title("Leave-one-EC3-class-out (60 classes)", fontweight='bold')
ax.legend(fontsize=6, loc='upper left')
ax.grid(axis='y', alpha=0.2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig1_benchmark.pdf")
plt.savefig(f"{PLOT_DIR}/fig1_benchmark.png", dpi=300)
print("Figure 1 saved", flush=True)

# ============================================================
# FIGURE 2: Interpretability
# ============================================================
with open(f"{OUT_DIR}/interpretability_results.json") as f:
    interp = json.load(f)

fig, axes = plt.subplots(2, 4, figsize=(8.5, 5.5))
axes = axes.flatten()

ec1_names = {
    "1": "EC1 Oxidoreductases",
    "2": "EC2 Transferases",
    "3": "EC3 Hydrolases",
    "4": "EC4 Lyases",
    "5": "EC5 Isomerases",
    "6": "EC6 Ligases",
    "7": "EC7 Translocases",
}

category_colors = {
    "Catalytic function": "#e41a1c",
    "Ligand-binding site": "#377eb8",
    "Structural motif": "#4daf4a",
    "Domain": "#984ea3",
    "Membrane-associated": "#ff7f00",
    "Disorder": "#ffff33",
    "Interaction site": "#a65628",
    "Compositional bias": "#f781bf",
    "Other": "#999999",
}

for c_idx, ec1_label in enumerate(sorted(interp["top_features_per_ec1"].keys(), key=lambda x: int(x))):
    ax = axes[c_idx]
    feats = interp["top_features_per_ec1"][ec1_label][:8]
    scores = [f["mi_score"] for f in feats]
    cats = [f.get("category", "Other") for f in feats]

    # Truncate long descriptions
    labels = []
    for f in feats:
        desc = f.get("description", f"F{f['feature_id']}")
        if len(desc) > 80:
            desc = desc[:77] + "..."
        labels.append(f"F{f['feature_id']}: {desc}")

    bar_colors = [category_colors.get(c, "#999999") for c in cats]

    ypos = range(len(feats))
    ax.barh(ypos, scores, color=bar_colors, edgecolor='white', linewidth=0.3, height=0.7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=5)
    ax.set_xlabel("MI", fontsize=6)
    ax.set_title(ec1_names.get(ec1_label, f"EC{ec1_label}"), fontsize=7, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend for categories
handles = [plt.Rectangle((0,0),1,1, color=c) for c in category_colors.values()]
labels = list(category_colors.keys())
axes[-1].legend(handles, labels, fontsize=5, loc='center', title="Feature Category",
                title_fontsize=6, ncol=2)
axes[-1].axis('off')

plt.suptitle("Top SAE Features Discriminating Each Enzyme Class", fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig2_interpretability.pdf")
plt.savefig(f"{PLOT_DIR}/fig2_interpretability.png", dpi=300)
print("Figure 2 saved", flush=True)

print("All figures generated!")
