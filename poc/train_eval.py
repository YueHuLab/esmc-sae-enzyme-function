"""
Train + stratified evaluation of feature representations for EC number prediction.

For each feature type (ESM-2 dense, SAE-like binary, k-mer 3):
  - 80/20 stratified split (by EC class)
  - Train HistGradientBoosting multi-class
  - For each test protein, compute max Jaccard to TRAINING set
  - Stratify by max-identity bins, report top-5 accuracy + AUROC
"""
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import top_k_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

OUT_DIR = "results"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# load
print("[1/4] loading features ...", flush=True)
emb_dense = np.load(f"{OUT_DIR}/esm2_8M_dense.npy")
sae_bin   = np.load(f"{OUT_DIR}/sae_like_binary.npy")
kmer3     = np.load(f"{OUT_DIR}/kmer3.npy")
J         = np.load(f"{OUT_DIR}/jaccard.npy")
with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
accs = meta["accs"]
ecs  = meta["ecs"]
le = LabelEncoder()
y = le.fit_transform(ecs)
n_classes = len(le.classes_)
print(f"  n_proteins={len(y)}  n_classes={n_classes}  features=dense:{emb_dense.shape}, sae:{sae_bin.shape}, kmer:{kmer3.shape}",
      flush=True)

# split
print("[2/4] splitting 80/20 stratified ...", flush=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(sss.split(np.zeros(len(y)), y))
y_tr, y_te = y[tr], y[te]
J_tr_tr = J[np.ix_(tr, tr)]
J_te_tr = J[np.ix_(te, tr)]                              # test × train jaccard

# for each test protein, max similarity to a TRAINING protein
max_sim = J_te_tr.max(axis=1)
print(f"  max-sim-to-train distribution: min={max_sim.min():.3f}, "
      f"q25={np.quantile(max_sim,0.25):.3f}, "
      f"median={np.median(max_sim):.3f}, "
      f"q75={np.quantile(max_sim,0.75):.3f}, "
      f"max={max_sim.max():.3f}", flush=True)

bins = [0.0, 0.20, 0.30, 0.40, 0.50, 0.65, 1.01]
bin_labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.65", ">=0.65"]
bin_idx = np.digitize(max_sim, bins[1:-1])

# ---------------------------------------------------------------------------
# 3. Train + evaluate per feature set
# ---------------------------------------------------------------------------
print("[3/4] training + per-bin evaluation ...", flush=True)
def train_eval(X_tr, X_te, name, use_lr=False):
    t0 = time.time()
    if use_lr:
        clf = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0, multi_class="multinomial")
    else:
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_depth=6, random_state=42)
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X_te)
    else:
        proba = clf.decision_function(X_te)
    # overall
    top1 = (proba.argmax(1) == y_te).mean()
    top5 = top_k_accuracy_score(y_te, proba, k=5, labels=np.arange(n_classes))
    auroc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro",
                          labels=np.arange(n_classes))
    print(f"  {name:25s}  top1={top1:.3f}  top5={top5:.3f}  AUROC={auroc:.3f}  "
          f"train={train_time:.1f}s", flush=True)

    # per-bin
    per_bin = []
    for b in range(len(bin_labels)):
        m = bin_idx == b
        if m.sum() == 0:
            per_bin.append({"n": 0, "top1": np.nan, "top5": np.nan})
            continue
        t1 = (proba[m].argmax(1) == y_te[m]).mean()
        t5 = top_k_accuracy_score(y_te[m], proba[m], k=5, labels=np.arange(n_classes))
        per_bin.append({"n": int(m.sum()), "top1": float(t1), "top5": float(t5)})
    return {"name": name, "top1": float(top1), "top5": float(top5), "auroc": float(auroc),
            "per_bin": per_bin, "train_time": train_time}

results = []
results.append(train_eval(emb_dense[tr], emb_dense[te], "ESM-2 8M dense"))
results.append(train_eval(sae_bin[tr],   sae_bin[te],   "SAE-like TopK=64 bin"))
results.append(train_eval(kmer3[tr],     kmer3[te],     "3-mer binary"))

# also linear probe for ESM-2 dense (sometimes a better baseline for embeddings)
results.append(train_eval(emb_dense[tr], emb_dense[te], "ESM-2 8M dense (LR)",
                          use_lr=True))

# ---------------------------------------------------------------------------
# 4. Save + plot
# ---------------------------------------------------------------------------
print("[4/4] saving results + plot ...", flush=True)
with open(f"{OUT_DIR}/benchmark.json", "w") as f:
    json.dump({"results": results, "bin_labels": bin_labels,
               "max_sim_distribution": {
                   "min": float(max_sim.min()),
                   "q25": float(np.quantile(max_sim, 0.25)),
                   "median": float(np.median(max_sim)),
                   "q75": float(np.quantile(max_sim, 0.75)),
                   "max": float(max_sim.max()),
               }}, f, indent=2)

# ---------- plot ----------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

# panel A: overall metrics
names = [r["name"] for r in results]
top1  = [r["top1"]  for r in results]
top5  = [r["top5"]  for r in results]
auroc = [r["auroc"] for r in results]
xpos = np.arange(len(names))
axes[0].bar(xpos - 0.25, top1,  width=0.25, label="Top-1 acc", color="#3b82f6")
axes[0].bar(xpos,         top5,  width=0.25, label="Top-5 acc", color="#10b981")
axes[0].bar(xpos + 0.25, auroc, width=0.25, label="AUROC (macro)", color="#f59e0b")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(names, rotation=20, ha="right", fontsize=9)
axes[0].set_ylabel("score")
axes[0].set_title("Overall EC prediction (50 classes, 968 train / 242 test)")
axes[0].legend(loc="lower right", fontsize=9)
axes[0].set_ylim(0, 1.0)
axes[0].grid(axis="y", alpha=0.3)

# panel B: top-5 acc vs. max jaccard bin
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
for i, r in enumerate(results):
    vals = [b["top5"] if b["n"] else np.nan for b in r["per_bin"]]
    axes[1].plot(range(len(bin_labels)), vals, "o-", color=colors[i % len(colors)],
                 label=r["name"], linewidth=2, markersize=7)
axes[1].set_xticks(range(len(bin_labels)))
axes[1].set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=9)
axes[1].set_xlabel("max 3-mer Jaccard to training set  (low = dark-matter-like)")
axes[1].set_ylabel("Top-5 accuracy")
axes[1].set_title("Accuracy vs. sequence-similarity to nearest training example")
axes[1].legend(loc="lower right", fontsize=8)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.0)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/benchmark.png", dpi=140, bbox_inches="tight")
print(f"  saved {PLOT_DIR}/benchmark.png", flush=True)

# table
print("\nPER-BIN Top-5 accuracy:")
print("bin           " + "  ".join(f"{r['name'][:18]:>18s}" for r in results))
for b, lab in enumerate(bin_labels):
    cells = []
    for r in results:
        v = r["per_bin"][b]["top5"]
        n = r["per_bin"][b]["n"]
        cells.append(f"{v:.3f} (n={n:3d})" if not np.isnan(v) else f"  --   (n={n:3d})")
    print(f"  {lab:10s}  " + "  ".join(f"{c:>18s}" for c in cells))
