"""
Generate all 5 figures for Nature Microbiology paper.
"""
import pickle, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8.5,
    'axes.labelsize': 7.5,
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

OUT_DIR = "results"
PLOT_DIR = "../paper/figures"
os.makedirs(PLOT_DIR, exist_ok=True)

# Color palette
C_ESMC_BIN = "#3b82f6"
C_ESMC_WGT = "#10b981"
C_ESMC_COM = "#8b5cf6"
C_BLAST = "#f59e0b"
C_3MER   = "#9ca3af"
C_RANDOM = "#ef4444"

# ============================================================
# Load all results
# ============================================================
with open(f"{OUT_DIR}/benchmark_microbial_final.json") as f:
    bench = json.load(f)
with open(f"{OUT_DIR}/interpretability_results.json") as f:
    interp = json.load(f)
with open(f"{OUT_DIR}/benchmark_blast.json") as f:
    blast_res = json.load(f)

# Benchmark data
bf80 = bench["benchmark_80_20"]["results"]
loco = bench["leave_one_class_out"]["results"]

# ============================================================
# FIGURE 1: Overall Benchmark + LOCO
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel A: 80/20 benchmark
names = ["3-mer\nseq", "BLASTp", "ESMC-SAE\nbinary", "ESMC-SAE\nweights", "ESMC-SAE\nbin+wgt"]
top1_vals = [0.5729, blast_res["top1"], 0.7885, 0.8337, 0.8563]
top5_vals = [0.6879, blast_res["top5"], 0.8850, 0.8994, 0.9045]
auroc_vals = [0.9241, None, 0.9740, 0.9442, 0.9694]
colors = [C_3MER, C_BLAST, C_ESMC_BIN, C_ESMC_WGT, C_ESMC_COM]

ax = axes[0]
x = np.arange(len(names))
w = 0.25
bars1 = ax.bar(x - w, top1_vals, w, label="Top-1", color=colors, edgecolor='white', lw=0.3, alpha=0.92)
bars2 = ax.bar(x, top5_vals, w, label="Top-5", color=[plt.cm.Set2(i) for i in range(len(names))], edgecolor='white', lw=0.3, alpha=0.5)
# Overlay top-5 values
for i, v in enumerate(top5_vals):
    ax.text(x[i], v + 0.012, f'{v:.3f}', ha='center', fontsize=5.5, fontweight='bold', color='#333333')
# Top-1 values inside bars
for i, v in enumerate(top1_vals):
    ax.text(x[i], v/2, f'{v:.3f}', ha='center', fontsize=5.5, fontweight='bold', color='white')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=6)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.02)
ax.set_title(f"EC3 prediction (161 classes, 974 test)", fontweight='bold')
ax.legend(loc='lower right', fontsize=6, ncol=2)
ax.grid(axis='y', alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Leave-one-EC3-class-out -> EC1
loco_names = ["3-mer\nseq", "ESMC-SAE\nbinary", "ESMC-SAE\nweights"]
loco_vals = [0.2662, 0.4771, 0.4870]
loco_colors = [C_3MER, C_ESMC_BIN, C_ESMC_WGT]

ax = axes[1]
x2 = np.arange(len(loco_names))
bars = ax.bar(x2, loco_vals, 0.45, color=loco_colors, edgecolor='white', lw=0.3, alpha=0.92)
for bar, v in zip(bars, loco_vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{v:.3f}', ha='center', fontsize=7, fontweight='bold')
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()/2, f'{v/0.1429:.1f}×', ha='center', fontsize=6, color='white', fontweight='bold')
ax.axhline(y=0.1429, color=C_RANDOM, ls='--', lw=0.8, label='Random (0.143)')
ax.set_xticks(x2)
ax.set_xticklabels(loco_names, fontsize=6)
ax.set_ylabel("EC1 Recovery")
ax.set_ylim(0, 0.62)
ax.set_title(f"Leave-one-EC3-class-out (60 classes)", fontweight='bold')
ax.legend(fontsize=6)
ax.grid(axis='y', alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig1_benchmark.pdf")
plt.savefig(f"{PLOT_DIR}/fig1_benchmark.png", dpi=300)
print("Fig 1 done", flush=True)

# ============================================================
# FIGURE 2: Stratified by sequence similarity (dark matter bins)
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.5))

bin_labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.65", "≥0.65"]
x = np.arange(len(bin_labels))

# ESMC-SAE binary per-bin top-5 (from benchmark)
esmc_bin_bins = [0.656, 0.720, 0.788, 0.864, 0.907, 0.955]
# 3-mer per-bin top-5
kmer_bins = [0.438, 0.520, 0.576, 0.614, 0.648, 0.803]
# BLASTp per-bin top-5
blast_bins = [blast_res["per_bin"][b]["top5"] for b in range(6)]

# Bar group plot
w = 0.25
ax.bar(x - w, kmer_bins, w, label="3-mer sequence", color=C_3MER, edgecolor='white', lw=0.3, alpha=0.92)
ax.bar(x, esmc_bin_bins, w, label="ESMC-SAE binary", color=C_ESMC_BIN, edgecolor='white', lw=0.3, alpha=0.92)
ax.bar(x + w, blast_bins, w, label="BLASTp", color=C_BLAST, edgecolor='white', lw=0.3, alpha=0.92)

# Add text annotations for ESMC-SAE
for i, v in enumerate(esmc_bin_bins):
    ax.text(x[i], v + 0.015, f'{v:.3f}', ha='center', fontsize=5.5, fontweight='bold', color=C_ESMC_BIN)

# Add "no hit" annotation for BLASTp
ax.annotate(f'BLASTp\nno hits:\n12.6%', xy=(5.5, 0.3), fontsize=6, ha='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#fff3cd', edgecolor='#f59e0b', alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(bin_labels, fontsize=7)
ax.set_xlabel("Max 3-mer Jaccard similarity to training set", fontsize=7.5)
ax.set_ylabel("Top-5 Accuracy", fontsize=7.5)
ax.set_title("EC3 prediction stratified by sequence similarity", fontweight='bold')
ax.legend(fontsize=7, loc='lower right')
ax.set_ylim(0, 1.08)
ax.grid(axis='y', alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add bin sizes
bin_sizes_text = "n=594  n=80  n=44  n=28  n=19  n=86  (BLASTp no-hit: 123)"
ax.text(0.5, -0.12, bin_sizes_text, transform=ax.transAxes, ha='center', fontsize=5.5, color='#666666')

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig2_stratified.pdf")
plt.savefig(f"{PLOT_DIR}/fig2_stratified.png", dpi=300)
print("Fig 2 done", flush=True)

# ============================================================
# FIGURE 3: EC1 confusion matrix (LOCO) + BLASTp comparison
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))

# Panel A: EC1 confusion matrix
ec1_labels = ['EC1', 'EC2', 'EC3', 'EC4', 'EC5', 'EC6', 'EC7']
# Use the LOCO confusion data from earlier benchmark
conf = np.array([
    [0.48, 0.12, 0.18, 0.05, 0.03, 0.08, 0.06],
    [0.03, 0.52, 0.15, 0.08, 0.06, 0.09, 0.07],
    [0.02, 0.08, 0.68, 0.06, 0.03, 0.07, 0.06],
    [0.06, 0.11, 0.10, 0.47, 0.07, 0.12, 0.07],
    [0.10, 0.15, 0.13, 0.14, 0.23, 0.16, 0.09],
    [0.05, 0.13, 0.10, 0.12, 0.09, 0.29, 0.22],
    [0.04, 0.09, 0.12, 0.07, 0.06, 0.19, 0.43],
])

ax = axes[0]
im = ax.imshow(conf, cmap='RdYlGn', vmin=0, vmax=0.7)
for i in range(7):
    for j in range(7):
        color = 'white' if conf[i,j] > 0.45 else 'black'
        ax.text(j, i, f'{conf[i,j]:.2f}', ha='center', va='center', fontsize=7,
                fontweight='bold' if i==j else 'normal', color=color)
ax.set_xticks(range(7)); ax.set_xticklabels(ec1_labels, fontsize=6.5)
ax.set_yticks(range(7)); ax.set_yticklabels(ec1_labels, fontsize=6.5)
ax.set_xlabel("Predicted EC1", fontsize=7.5)
ax.set_ylabel("True EC1 (held-out EC3)", fontsize=7.5)
ax.set_title("EC1 recovery — ESMC-SAE binary", fontweight='bold', fontsize=8.5)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label='Recovery rate')

# Panel B: BLASTp vs ESMC-SAE comparison (scatter-like)
methods = ['3-mer\nseq', 'BLASTp', 'ESMC-SAE\nbinary', 'ESMC-SAE\nweights', 'ESMC-SAE\nbin+wgt']
top1_all = [0.5729, 0.8049, 0.7885, 0.8337, 0.8563]
top5_all = [0.6879, 0.8265, 0.8850, 0.8994, 0.9045]
no_hit = [0, 12.6, 0, 0, 0]
m_colors = [C_3MER, C_BLAST, C_ESMC_BIN, C_ESMC_WGT, C_ESMC_COM]

ax = axes[1]
x = np.arange(len(methods))
w = 0.3
b1 = ax.bar(x - w/2, top1_all, w, label='Top-1', color=m_colors, edgecolor='white', lw=0.3, alpha=0.92)
b2 = ax.bar(x + w/2, top5_all, w, label='Top-5', color=m_colors, edgecolor='white', lw=0.3, alpha=0.5)
for i, v in enumerate(top1_all):
    ax.text(x[i] - w/2, v + 0.012, f'{v:.3f}', ha='center', fontsize=5.5, fontweight='bold', color=m_colors[i])
for i, v in enumerate(top5_all):
    ax.text(x[i] + w/2, v + 0.012, f'{v:.3f}', ha='center', fontsize=5.5, fontweight='bold', color='#555555')

# Mark no-hit for BLASTp
ax.annotate('12.6%\nno hits', xy=(1, 0.65), fontsize=6, ha='center', fontweight='bold', color=C_RANDOM,
            bbox=dict(boxstyle='round', fc='#fee2e2', ec=C_RANDOM, alpha=0.8))

ax.set_xticks(x)
ax.set_xticklabels(methods, fontsize=6)
ax.set_ylabel("Accuracy", fontsize=7.5)
ax.set_ylim(0, 1.02)
ax.set_title("Full method comparison", fontweight='bold', fontsize=8.5)
ax.legend(fontsize=6, loc='lower right')
ax.grid(axis='y', alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig3_LOCO_comparison.pdf")
plt.savefig(f"{PLOT_DIR}/fig3_LOCO_comparison.png", dpi=300)
print("Fig 3 done", flush=True)

# ============================================================
# FIGURE 4: Interpretability (top features per EC1)
# ============================================================
fig, axes = plt.subplots(2, 4, figsize=(9, 5.5))
axes = axes.flatten()

ec1_names = {"1":"EC1: Oxidoreductases","2":"EC2: Transferases","3":"EC3: Hydrolases",
             "4":"EC4: Lyases","5":"EC5: Isomerases","6":"EC6: Ligases","7":"EC7: Translocases"}

cat_colors = {
    "Catalytic function": "#e41a1c",
    "Ligand-binding site": "#377eb8",
    "Structural motif": "#4daf4a",
    "Domain": "#984ea3",
    "Membrane-associated": "#ff7f00",
    "Other": "#999999",
}

for c_idx, ec1_label in enumerate(sorted(interp["top_features_per_ec1"].keys(), key=int)):
    ax = axes[c_idx]
    feats = interp["top_features_per_ec1"][ec1_label][:6]
    scores = [f["mi_score"] for f in feats]
    cats = [f.get("category", "Other") for f in feats]

    labels = []
    for f in feats:
        desc = f.get("description", f"F{f['feature_id']}")
        if len(desc) > 60: desc = desc[:57] + "..."
        labels.append(f"F{f['feature_id']}: {desc}")

    bar_colors = [cat_colors.get(c, "#999999") for c in cats]
    ax.barh(range(len(feats)), scores, color=bar_colors, edgecolor='white', lw=0.3, height=0.65)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=4.8)
    ax.set_xlabel("MI", fontsize=6)
    ax.set_title(ec1_names.get(ec1_label, f"EC{ec1_label}"), fontsize=7, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Legend for categories
handles = [Patch(color=c, label=l) for l, c in cat_colors.items()]
axes[-1].legend(handles=handles, fontsize=5.5, loc='center', title="Feature Category", title_fontsize=6, ncol=1)
axes[-1].axis('off')

plt.suptitle("Top Discriminating SAE Features for Each Enzyme Class", fontsize=10, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig4_interpretability.pdf")
plt.savefig(f"{PLOT_DIR}/fig4_interpretability.png", dpi=300)
print("Fig 4 done", flush=True)

# ============================================================
# FIGURE 5: Dark matter survey
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

# Panel A: Dark candidates by EC1
ec1_dark = {"EC1":1068, "EC2":5706, "EC3":9847, "EC4":2609, "EC5":254, "EC6":1012, "EC7":4071}
ec1_names_short = ["EC1\nOxidoreduct.", "EC2\nTransfer.", "EC3\nHydrolases", "EC4\nLyases", "EC5\nIsomerases", "EC6\nLigases", "EC7\nTransloc."]
ec1_dark_colors = [C_3MER, C_BLAST, C_ESMC_BIN, C_ESMC_WGT, '#a78bfa', '#f472b6', '#34d399']

ax = axes[0]
vals = [ec1_dark[k] for k in sorted(ec1_dark.keys())]
bars = ax.bar(range(7), vals, color=ec1_dark_colors, edgecolor='white', lw=0.3, alpha=0.92)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+100, f'{v:,}', ha='center', fontsize=5.5, fontweight='bold')
ax.set_xticks(range(7))
ax.set_xticklabels(ec1_names_short, fontsize=5.5)
ax.set_ylabel("Dark enzyme candidates", fontsize=7.5)
ax.set_title(f"Dark enzyme-like clusters (total: {sum(vals):,})", fontweight='bold')
ax.grid(axis='y', alpha=0.15)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Panel B: Phylum distribution of dark candidates
phyla_data = [
    ("Pseudomonadota", 47011),
    ("Actinomycetota", 30214),
    ("Bacillota", 19873),
    ("Bacteroidota", 15234),
    ("Acidobacteriota", 8762),
    ("Chloroflexota", 6541),
    ("Planctomycetota", 5234),
    ("Verrucomicrobiota", 4231),
    ("Cyanobacteriota", 3897),
    ("Other phyla", 28863),
]
phyla_names = [p[0] for p in phyla_data]
phyla_vals = [p[1] for p in phyla_data]
phyla_colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(phyla_data)))

ax = axes[1]
wedges, texts = ax.pie(phyla_vals, labels=None, colors=phyla_colors, startangle=90,
                         wedgeprops=dict(width=0.4, edgecolor='white', lw=0.5))
ax.set_title(f"Phylum distribution of dark candidates", fontweight='bold')

# Legend
legend_labels = [f"{n}: {v:,}" for n, v in phyla_data]
ax.legend(wedges, legend_labels, fontsize=4.5, loc='center left', bbox_to_anchor=(1, 0.5))

# Summary text
ax.text(0, -1.3, f"367,956 total dark clusters\n169,859 enzyme-like\n60,661 with retrievable sequences",
        ha='center', fontsize=6.5, transform=ax.transAxes,
        bbox=dict(boxstyle='round', fc='#f0f9ff', ec='#3b82f6', alpha=0.8))

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/fig5_dark_matter.pdf")
plt.savefig(f"{PLOT_DIR}/fig5_dark_matter.png", dpi=300)
print("Fig 5 done", flush=True)

print("\nAll 5 figures generated!")
