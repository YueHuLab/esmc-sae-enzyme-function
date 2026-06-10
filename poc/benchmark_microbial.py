"""
Full benchmark on microbial enzyme dataset: 80/20 stratified + leave-one-EC-class-out.
Predicts EC3 (e.g., 1.1.1) for robust class sizes; evaluates EC1 for leave-class-out.

ALL Linear Probe (LogisticRegression).

Compares:
  1. 3-mer binary (8000d) — sequence baseline
  2. ESMC-SAE binary (16384d) — our method
  3. ESMC-SAE weights (16384d) — our method (raw)
  4. ESMC-SAE binary+weights combo (32768d)

Key metrics:
  - Top-1 / Top-5 accuracy (80/20, stratified by EC3)
  - Per-bin accuracy (6 sequence-similarity bins, <0.20 = "dark matter")
  - Leave-one-EC3-class-out → EC1 accuracy
  - AUROC (macro OVR)
"""
import os, json, time, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import top_k_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer

OUT_DIR = "results"
PLOT_DIR = "plots"
PREFIX = "microbial_esmc_sae"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---- load features ----
print("[1/5] Loading features ...", flush=True)
weights = np.load(f"{OUT_DIR}/{PREFIX}_weights.npy")
binary = np.load(f"{OUT_DIR}/{PREFIX}_binary.npy")

with open(f"{OUT_DIR}/{PREFIX}_meta.pkl", "rb") as f:
    meta = pickle.load(f)

ecs = meta["ecs"]      # full EC (e.g., "1.1.1.1")
ec1s = meta["ec1s"]
n_proteins = len(ecs)

# Use EC3 (first 3 levels) as prediction target for robust class sizes
ec3s = [".".join(ec.split(".")[:3]) for ec in ecs]
unique_ec3 = sorted(set(ec3s))
print(f"  EC3 classes: {len(unique_ec3)}")
# Filter to EC3 classes with >= 5 members for meaningful evaluation
ec3_counts = {}
for ec3 in ec3s:
    ec3_counts[ec3] = ec3_counts.get(ec3, 0) + 1
valid_ec3 = {ec3 for ec3, cnt in ec3_counts.items() if cnt >= 5}

print(f"  EC3 classes with >=5 members: {len(valid_ec3)}")

# Filter proteins to those in valid EC3 classes
keep_idx = [i for i, ec3 in enumerate(ec3s) if ec3 in valid_ec3]
print(f"  Proteins in valid classes: {len(keep_idx)}")

weights_f = weights[keep_idx]
binary_f = binary[keep_idx]
ec3s_f = [ec3s[i] for i in keep_idx]
ec1s_f = [ec1s[i] for i in keep_idx]
n = len(keep_idx)

# Encode EC3
le = LabelEncoder()
y = le.fit_transform(ec3s_f)
n_classes = len(le.classes_)

# Encode EC1
ec1_le = LabelEncoder()
y_ec1 = ec1_le.fit_transform(ec1s_f)
n_ec1 = len(ec1_le.classes_)

print(f"  Final: {n} proteins, {n_classes} EC3 classes, {n_ec1} EC1 classes")
print(f"  EC1 distribution: {dict(zip(ec1_le.classes_, np.bincount(y_ec1)))}", flush=True)

# ---- compute 3-mer Jaccard for binning ----
print("[2/5] Computing 3-mer Jaccard for similarity binning...", flush=True)

# Load sequences in same order as features
seqs = []
with open("data/microbial_enzymes_5k.tsv") as f:
    header = f.readline().strip().split("\t")
    acc_idx = header.index("Entry")
    seq_idx = header.index("Sequence")
    # Build accession → sequence lookup
    acc2seq = {}
    for line in f:
        row = line.strip().split("\t")
        if len(row) > seq_idx and len(row) > acc_idx:
            acc2seq[row[acc_idx]] = row[seq_idx]

# Load in meta accession order (same as feature order)
for acc in meta["accs"]:
    if acc in acc2seq:
        seqs.append(acc2seq[acc])
    else:
        seqs.append("")  # placeholder

assert len(seqs) == n_proteins, f"Seq count {len(seqs)} != protein count {n_proteins}"
seqs_f = [seqs[i] for i in keep_idx]

vectorizer = CountVectorizer(analyzer="char", ngram_range=(3, 3), lowercase=False,
                             max_features=8000, binary=True)
kmer_matrix = vectorizer.fit_transform(seqs_f).toarray().astype(np.uint8)
print(f"  3-mer matrix: {kmer_matrix.shape}", flush=True)

# ---- 80/20 split ----
print("[3/5] 80/20 split + training (all LR)...", flush=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(sss.split(np.zeros(n), y))
y_tr, y_te = y[tr], y[te]

print(f"  Train: {len(tr)}, Test: {len(te)}", flush=True)

# Compute Jaccard distances for test-train binning
kmer_tr = kmer_matrix[tr]
kmer_te = kmer_matrix[te]

# Use sampling for speed: max 2000 comparisons per test protein
import random
random.seed(42)
max_comparisons = min(2000, len(tr))
sample_tr = random.sample(range(len(tr)), max_comparisons)
kmer_tr_sample = kmer_tr[sample_tr]

max_sim = np.zeros(len(te))
for i in range(len(te)):
    a = kmer_te[i]
    inter = (a & kmer_tr_sample).sum(axis=1)
    union = (a | kmer_tr_sample).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        jacc = np.where(union > 0, inter / union, 0)
    max_sim[i] = jacc.max()

print(f"  Max 3-mer Jaccard: min={max_sim.min():.3f}, median={np.median(max_sim):.3f}, "
      f"max={max_sim.max():.3f}", flush=True)

bins = [0.0, 0.20, 0.30, 0.40, 0.50, 0.65, 1.01]
bin_labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.65", ">=0.65"]
bin_idx = np.digitize(max_sim, bins[1:-1])

# ---- train & evaluate ----
def train_eval(X_tr, X_te, name, scale=False):
    t0 = time.time()
    if scale:
        sc = StandardScaler().fit(X_tr)
        X_tr, X_te = sc.transform(X_tr), sc.transform(X_te)

    clf = LogisticRegression(max_iter=500, n_jobs=-1, C=1.0)
    clf.fit(X_tr, y_tr)
    train_time = time.time() - t0
    proba = clf.predict_proba(X_te)

    top1 = (proba.argmax(1) == y_te).mean()
    top5 = top_k_accuracy_score(y_te, proba, k=5, labels=np.arange(n_classes))
    auroc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro",
                          labels=np.arange(n_classes))

    per_bin = []
    for b in range(len(bin_labels)):
        m = bin_idx == b
        if m.sum() == 0:
            per_bin.append({"n": 0, "top1": float('nan'), "top5": float('nan')})
            continue
        probs_m = proba[m]
        y_m = y_te[m]
        t1 = (probs_m.argmax(1) == y_m).mean()
        t5 = top_k_accuracy_score(y_m, probs_m, k=5, labels=np.arange(n_classes))
        per_bin.append({"n": int(m.sum()), "top1": float(t1), "top5": float(t5)})

    print(f"  {name:35s}  top1={top1:.4f}  top5={top5:.4f}  "
          f"AUROC={auroc:.4f}  {train_time:.1f}s", flush=True)
    return {"name": name, "top1": float(top1), "top5": float(top5),
            "auroc": float(auroc), "per_bin": per_bin, "train_time": train_time}

print("  --- 80/20 benchmark (EC3 prediction) ---")
results = []
results.append(train_eval(kmer_matrix[tr], kmer_matrix[te], "3-mer binary (8000d)"))
results.append(train_eval(binary_f[tr], binary_f[te], "ESMC-SAE binary (16384d)"))
results.append(train_eval(weights_f[tr], weights_f[te], "ESMC-SAE weights (16384d)", scale=True))
combined = np.concatenate([binary_f.astype(np.float32),
                           StandardScaler().fit_transform(weights_f)], axis=1)
results.append(train_eval(combined[tr], combined[te], "ESMC-SAE binary+weights (32768d)"))

# ---- leave-one-EC3-class-out → EC1 evaluation ----
print("\n  --- Leave-one-EC3-class-out (EC1 accuracy) ---")

def leave_one_out(X, name, scale=False, max_classes=80):
    """Leave-one-EC3-class-out → EC1 accuracy. Limited to max_classes for speed."""
    t0 = time.time()

    # Select classes with most members for evaluation
    class_sizes = [(c, (y == c).sum()) for c in range(n_classes)]
    class_sizes.sort(key=lambda x: -x[1])
    eval_classes = [c for c, sz in class_sizes[:max_classes] if sz >= 5]

    n_eval = 0
    ec1_acc_sum = 0.0
    ec1_conf = np.zeros((n_ec1, n_ec1), dtype=int)

    for c in eval_classes:
        mask_te = y == c
        idx_te = np.where(mask_te)[0]
        idx_tr = np.where(~mask_te)[0]
        if len(idx_te) < 3 or len(idx_tr) < 200:
            continue
        Xtr, Xte = X[idx_tr], X[idx_te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        clf = LogisticRegression(max_iter=300, n_jobs=-1, C=1.0, solver='lbfgs')
        clf.fit(Xtr, y[idx_tr])
        proba = clf.predict_proba(Xte)
        global_proba = np.full((len(idx_te), n_classes), 1e-9, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            global_proba[:, cls] = proba[:, j]
        pred = global_proba.argmax(1)
        pred_ec1_raw = [le.classes_[p].split(".")[0] for p in pred]
        pred_ec1 = ec1_le.transform(pred_ec1_raw)
        true_ec1 = y_ec1[idx_te]
        for t, p in zip(true_ec1, pred_ec1):
            ec1_conf[t, p] += 1

        n_eval += 1
        ec1_acc_sum += (pred_ec1 == true_ec1).mean()
        if n_eval % 20 == 0:
            print(f"    {n_eval}/{len(eval_classes)}, EC1_acc={ec1_acc_sum/n_eval:.4f}", flush=True)

    ec1_acc = ec1_acc_sum / n_eval
    print(f"  {name:35s}  EC1={ec1_acc:.4f}  ({n_eval} classes, {time.time()-t0:.1f}s)", flush=True)
    return ec1_acc, ec1_conf, n_eval

loco_results = []
best_conf, best_name, best_n = None, "", 0

for X, name, scale in [
    (kmer_matrix, "3-mer binary (8000d)", False),
    (binary_f, "ESMC-SAE binary (16384d)", False),
    (weights_f, "ESMC-SAE weights (16384d)", True),
]:
    ec1, conf, n_ev = leave_one_out(X, name, scale=scale)
    loco_results.append({"name": name, "ec1_acc": ec1, "n_eval": n_ev})
    if "ESMC-SAE binary" in name:
        best_conf, best_name, best_n = conf, name, n_ev

# ---- save results ----
print("[4/5] Saving results...", flush=True)
output = {
    "overall_80_20": results,
    "leave_one_class_out": loco_results,
    "bin_labels": bin_labels,
    "bin_counts": [int((bin_idx == b).sum()) for b in range(len(bin_labels))],
    "random_ec1_baseline": 1.0 / n_ec1,
    "n_proteins": n,
    "n_ec3_classes": n_classes,
    "n_ec1_classes": n_ec1,
    "ec1_distribution": {str(k): int(v) for k, v in zip(ec1_le.classes_, np.bincount(y_ec1))},
    "dataset": "microbial_swissprot_5k",
    "prediction_level": "EC3",
}

with open(f"{OUT_DIR}/benchmark_microbial.json", "w") as f:
    json.dump(output, f, indent=2)

# ---- plots ----
print("[5/5] Generating figures...", flush=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(21, 5.5))

# Panel A: overall bar chart
names = [r["name"] for r in results]
top1 = [r["top1"] for r in results]
top5 = [r["top5"] for r in results]
auroc = [r["auroc"] for r in results]
xpos = np.arange(len(names))
short = [n.replace("ESMC-SAE ", "ESMC-SAE\n").replace(" binary", " bin")
          .replace(" weights", " wgt").replace(" binary+weights", " bin+wgt")
          .replace(" (", "\n(").replace("3-mer binary", "3-mer") for n in names]

axes[0].bar(xpos - 0.25, top1, width=0.25, label="Top-1", color="#3b82f6")
axes[0].bar(xpos, top5, width=0.25, label="Top-5", color="#10b981")
axes[0].bar(xpos + 0.25, auroc, width=0.25, label="AUROC", color="#f59e0b")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(short, rotation=0, ha="center", fontsize=7)
axes[0].set_ylabel("Score")
axes[0].set_title(f"EC3 prediction — {n_classes} classes, {n} microbial enzymes")
axes[0].legend(loc="lower right", fontsize=7)
axes[0].set_ylim(0, 1.05)
axes[0].grid(axis="y", alpha=0.3)

# Panel B: top-5 vs sequence similarity
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]
markers = ["o", "s", "^", "D"]
for i, r in enumerate(results):
    vals = []
    for b in r["per_bin"]:
        if b["n"] > 0 and not np.isnan(b.get("top5", float('nan'))):
            vals.append(b["top5"])
        else:
            vals.append(np.nan)
    axes[1].plot(range(len(bin_labels)), vals, marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=r["name"][:40], linewidth=2, markersize=7)
axes[1].set_xticks(range(len(bin_labels)))
axes[1].set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=9)
axes[1].set_xlabel("Max 3-mer Jaccard to training set")
axes[1].set_ylabel("Top-5 accuracy")
n_test = len(te)
axes[1].set_title(f"Accuracy vs sequence similarity ({n_test} test proteins, dark bins = left)")
axes[1].legend(loc="lower right", fontsize=6)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

# Panel C: EC1 confusion matrix
if best_conf is not None:
    conf_norm = best_conf / (best_conf.sum(axis=1, keepdims=True) + 1e-9)
    im = axes[2].imshow(conf_norm, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_xticks(range(n_ec1))
    axes[2].set_xticklabels(ec1_le.classes_)
    axes[2].set_yticks(range(n_ec1))
    axes[2].set_yticklabels(ec1_le.classes_)
    axes[2].set_xlabel("Predicted EC1")
    axes[2].set_ylabel("True EC1 (held-out EC3)")
    axes[2].set_title(f"EC1 recovery — {best_name.split(chr(32))[0]}\n"
                      f"(leave-EC3-out, {best_n} classes)")
    for i in range(n_ec1):
        for j in range(n_ec1):
            axes[2].text(j, i, f"{conf_norm[i,j]:.2f}", ha="center", va="center",
                         color="white" if conf_norm[i, j] > 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/benchmark_microbial.png", dpi=150, bbox_inches="tight")
print(f"  Saved {PLOT_DIR}/benchmark_microbial.png", flush=True)

# ---- summary ----
print("\n" + "=" * 90)
print("MICROBIAL ENZYME EC3 PREDICTION — FULL BENCHMARK")
print("=" * 90)
print(f"{'Feature':40s}  {'Top-1':>6s}  {'Top-5':>6s}  {'AUROC':>6s}  {'EC1-LOCO':>9s}  {'Time':>6s}")
print("-" * 90)
for r in results:
    loco_match = [l for l in loco_results if l["name"] == r["name"]]
    loco_str = f"{loco_match[0]['ec1_acc']:.4f}" if loco_match else "  --"
    print(f"{r['name']:40s}  {r['top1']:6.4f}  {r['top5']:6.4f}  {r['auroc']:6.4f}  {loco_str:>9s}  {r['train_time']:5.1f}s")

print(f"\nRandom EC1 baseline: {1/n_ec1:.4f}")
print(f"Bin sizes: {dict(zip(bin_labels, [int((bin_idx==b).sum()) for b in range(len(bin_labels))]))}")
print(f"Saved to {OUT_DIR}/benchmark_microbial.json")
