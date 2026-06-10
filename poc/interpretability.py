"""
SAE feature interpretability analysis for EC prediction.
For each EC1 class, find top discriminating SAE features and map them to
GPT-5 biological descriptions from ESMC-SAE-Features.

Produces:
  - EC1 vs feature association table
  - Top features per EC class with biological interpretations
  - Figure for paper
"""
import os, pickle, json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif

OUT_DIR = "results"
PLOT_DIR = "plots"
PREFIX = "microbial_esmc_sae"
FEATURE_DIR = "/Users/huyue/esm-c-fold2/ESMC-SAE-Features"
FEATURE_DIM = 16384

os.makedirs(PLOT_DIR, exist_ok=True)

# ---- Load data ----
print("[1/5] Loading features and descriptions...", flush=True)
weights = np.load(f"{OUT_DIR}/{PREFIX}_weights.npy")
binary = np.load(f"{OUT_DIR}/{PREFIX}_binary.npy")
with open(f"{OUT_DIR}/{PREFIX}_meta.pkl", "rb") as f:
    meta = pickle.load(f)

ec1s = meta["ec1s"]
n_proteins = len(ec1s)

# Encode EC1
ec1_le = LabelEncoder()
y_ec1 = ec1_le.fit_transform(ec1s)
n_ec1 = len(ec1_le.classes_)
print(f"  Proteins: {n_proteins}, EC1 classes: {n_ec1}")
print(f"  EC1: {dict(zip(ec1_le.classes_, np.bincount(y_ec1)))}", flush=True)

# ---- Load feature descriptions ----
print("[2/5] Loading feature descriptions...", flush=True)
import pandas as pd
desc_files = [f for f in os.listdir(FEATURE_DIR) if f.endswith('.parquet')]
if desc_files:
    desc_df = pd.read_parquet(os.path.join(FEATURE_DIR, desc_files[0]))
    print(f"  Loaded descriptions: {desc_df.shape}")
    print(f"  Columns: {desc_df.columns.tolist()}")
    # Try to identify feature_id and description columns
    print(f"  Sample:\n{desc_df.head(2).to_string()}")
else:
    print("  No description files found, using feature indices only")
    desc_df = None

# ---- Feature importance by EC1 ----
print("[3/5] Computing feature importance per EC1 class...", flush=True)

# Use mutual information (fast, interpretable)
mi_scores = np.zeros((n_ec1, FEATURE_DIM))
for c in range(n_ec1):
    y_binary = (y_ec1 == c).astype(int)
    mi = mutual_info_classif(binary, y_binary, random_state=42, n_neighbors=5)
    mi_scores[c] = mi
    print(f"  EC{ec1_le.classes_[c]}: MI computed", flush=True)

# Top features per EC1
TOP_N = 20
ec1_names = {
    "1": "Oxidoreductases",
    "2": "Transferases",
    "3": "Hydrolases",
    "4": "Lyases",
    "5": "Isomerases",
    "6": "Ligases",
    "7": "Translocases",
}

top_features = {}
for c in range(n_ec1):
    ec1_label = ec1_le.classes_[c]
    top_idx = np.argsort(-mi_scores[c])[:TOP_N]
    top_features[ec1_label] = []
    for rank, feat_id in enumerate(top_idx):
        ec1_name = ec1_names.get(ec1_label, f"EC{ec1_label}")
        feat_info = {
            "feature_id": int(feat_id),
            "mi_score": float(mi_scores[c, feat_id]),
            "description": "N/A",
            "category": "N/A",
        }
        top_features[ec1_label].append(feat_info)
    print(f"  EC{ec1_label} ({ec1_name}): top feature={top_features[ec1_label][0]['feature_id']}, "
          f"MI={top_features[ec1_label][0]['mi_score']:.4f}", flush=True)

# ---- Feature descriptions (if available) ----
print("[4/5] Matching features to biological descriptions...", flush=True)

# Build feature description lookup (known schema: feature_id, summary, category)
feature_desc = {}
feature_cat = {}
if desc_df is not None and "feature_id" in desc_df.columns and "summary" in desc_df.columns:
    for _, row in desc_df.iterrows():
        fid = int(row["feature_id"])
        feature_desc[fid] = str(row["summary"])[:200]  # truncate for display
        feature_cat[fid] = str(row.get("category", "Unknown"))
    print(f"  Loaded {len(feature_desc)} feature descriptions")
elif not feature_desc:
    print("  No description mapping found, using placeholder labels")
    # Generate placeholder based on feature index
    feature_desc = {i: f"Feature_{i}" for i in range(FEATURE_DIM)}
    feature_cat = {i: "Unknown" for i in range(FEATURE_DIM)}

# Update top_features with descriptions
for ec1_label in top_features:
    for feat_info in top_features[ec1_label]:
        fid = feat_info["feature_id"]
        feat_info["description"] = feature_desc.get(fid, f"Feature_{fid}")
        feat_info["category"] = feature_cat.get(fid, "Unknown")

# ---- Generate figure ----
print("[5/5] Generating interpretability figure...", flush=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for c in range(n_ec1):
    ax = axes[c]
    ec1_label = ec1_le.classes_[c]
    ec1_name = ec1_names.get(ec1_label, f"EC{ec1_label}")

    feats = top_features[ec1_label][:10]
    scores = [f["mi_score"] for f in feats]
    labels = [f"F{f['feature_id']}: {f['description'][:60]}" for f in feats]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feats)))
    ax.barh(range(len(feats)), scores, color=colors)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Mutual Information")
    ax.set_title(f"EC{ec1_label} — {ec1_name}", fontsize=10, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

# Hide extra subplot if n_ec1 < 8
if n_ec1 < 8:
    axes[-1].set_visible(False)

plt.suptitle("Top SAE Features Discriminating Each EC1 Class\n"
             f"(mutual information, {n_proteins} microbial enzymes, {FEATURE_DIM} SAE features)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/interpretability_ec1.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR}/interpretability_ec1.png", flush=True)

# ---- Cross-EC feature overlap analysis ----
# Find features that are important for MULTIPLE EC classes (shared concepts)
print("\n  Shared features across EC classes:")
shared = {}
for fid in range(FEATURE_DIM):
    ecs_with = []
    for c in range(n_ec1):
        if fid in [f["feature_id"] for f in top_features[ec1_le.classes_[c]]]:
            ecs_with.append(ec1_le.classes_[c])
    if len(ecs_with) >= 2:
        shared[fid] = ecs_with

if shared:
    print(f"  {len(shared)} features shared across 2+ EC classes")
    for fid, ecs in sorted(shared.items(), key=lambda x: -len(x[1]))[:10]:
        desc = feature_desc.get(fid, f"Feature_{fid}")
        print(f"    F{fid}: shared by EC{','.join(ecs)} — {desc[:100]}")

# ---- Save ----
output = {
    "n_proteins": n_proteins,
    "n_ec1": n_ec1,
    "ec1_classes": list(ec1_le.classes_),
    "top_features_per_ec1": top_features,
    "shared_features": {str(k): {"ec_classes": v, "description": feature_desc.get(k, "")}
                        for k, v in shared.items()},
}

with open(f"{OUT_DIR}/interpretability_results.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\nSaved to {OUT_DIR}/interpretability_results.json")
print("Done!")
