"""
Compare real ESMC-SAE features vs simulated SAE vs baseline.
ALL linear probe (LogisticRegression) — standard for pre-trained feature eval.
Binary features: no scaling. Dense features: StandardScaler.
"""
import os, json, time, pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import top_k_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

OUT_DIR = "results"
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ---- load ----
print("[1/3] Loading features ...", flush=True)
emb_dense = np.load(f"{OUT_DIR}/esm2_8M_dense.npy")
sae_sim   = np.load(f"{OUT_DIR}/sae_like_binary.npy")
kmer3     = np.load(f"{OUT_DIR}/kmer3.npy")
esmc_w    = np.load(f"{OUT_DIR}/esmc_sae_weights.npy")
esmc_b    = np.load(f"{OUT_DIR}/esmc_sae_binary.npy")
J         = np.load(f"{OUT_DIR}/jaccard.npy")

with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
ecs = meta["ecs"]; ec1s = meta["ec1"]
le = LabelEncoder(); y = le.fit_transform(ecs)
n_classes = len(le.classes_)
ec1_le = LabelEncoder(); y_ec1 = ec1_le.fit_transform(ec1s)
n_ec1 = len(ec1_le.classes_)
n_proteins = len(y)
print(f"  n={n_proteins}  classes={n_classes}  ec1={n_ec1}", flush=True)

# ---- 80/20 split ----
print("[2/3] 80/20 split + training (all LR)...", flush=True)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(sss.split(np.zeros(len(y)), y))
y_tr, y_te = y[tr], y[te]
J_te_tr = J[np.ix_(te, tr)]
max_sim = J_te_tr.max(axis=1)

bins = [0.0, 0.20, 0.30, 0.40, 0.50, 0.65, 1.01]
bin_labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.65", ">=0.65"]
bin_idx = np.digitize(max_sim, bins[1:-1])

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
    auroc = roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=np.arange(n_classes))
    per_bin = []
    for b in range(len(bin_labels)):
        m = bin_idx == b
        if m.sum() == 0:
            per_bin.append({"n": 0, "top1": float('nan'), "top5": float('nan')})
            continue
        t1 = (proba[m].argmax(1) == y_te[m]).mean()
        t5 = top_k_accuracy_score(y_te[m], proba[m], k=5, labels=np.arange(n_classes))
        per_bin.append({"n": int(m.sum()), "top1": float(t1), "top5": float(t5)})
    print(f"  {name:35s}  top1={top1:.3f}  top5={top5:.3f}  AUROC={auroc:.3f}  {train_time:.1f}s", flush=True)
    return {"name": name, "top1": float(top1), "top5": float(top5), "auroc": float(auroc),
            "per_bin": per_bin, "train_time": train_time}

print("  --- 80/20 benchmark ---")
results = []
results.append(train_eval(emb_dense[tr], emb_dense[te], "ESM-2 8M dense (320d)", scale=True))
results.append(train_eval(sae_sim[tr],   sae_sim[te],   "Simulated SAE TopK=64 (320d)"))
results.append(train_eval(kmer3[tr],     kmer3[te],     "3-mer binary (8000d)"))
results.append(train_eval(esmc_b[tr],    esmc_b[te],    "REAL ESMC-SAE binary (16384d)"))
results.append(train_eval(esmc_w[tr],    esmc_w[te],    "REAL ESMC-SAE weights (16384d)", scale=True))

# ---- leave-one-EC-class-out ----
print("\n  --- Leave-one-EC-class-out ---")

def leave_one_out(X, name, scale=False):
    t0 = time.time()
    n_eval = 0
    ec1_acc_sum = 0.0
    ec1_conf = np.zeros((n_ec1, n_ec1), dtype=int)
    for c in range(n_classes):
        mask_te = y == c
        idx_te = np.where(mask_te)[0]; idx_tr = np.where(~mask_te)[0]
        if len(idx_te) < 3 or len(idx_tr) < 100:
            continue
        Xtr = X[idx_tr]; Xte = X[idx_te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=500, n_jobs=-1, C=1.0)
        clf.fit(Xtr, y[idx_tr])
        proba = clf.predict_proba(Xte)
        global_proba = np.full((len(idx_te), n_classes), 1e-9, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            global_proba[:, cls] = proba[:, j]
        pred = global_proba.argmax(1)
        pred_ec1 = ec1_le.transform([le.classes_[p].split(".")[0] for p in pred])
        true_ec1 = y_ec1[idx_te]
        for t, p in zip(true_ec1, pred_ec1):
            ec1_conf[t, p] += 1
        n_eval += 1
        ec1_acc_sum += (pred_ec1 == true_ec1).sum() / len(idx_te)
    ec1_acc = ec1_acc_sum / n_eval
    print(f"  {name:35s}  EC1={ec1_acc:.3f}  ({n_eval} classes, {time.time()-t0:.1f}s)", flush=True)
    return ec1_acc, ec1_conf, n_eval

loco_results = []
best_conf, best_name = None, ""

for X, name, scale in [
    (emb_dense, "ESM-2 8M dense (320d)", True),
    (sae_sim,   "Simulated SAE TopK=64 (320d)", False),
    (kmer3,     "3-mer binary (8000d)", False),
    (esmc_b,    "REAL ESMC-SAE binary (16384d)", False),
    (esmc_w,    "REAL ESMC-SAE weights (16384d)", True),
]:
    ec1, conf, n = leave_one_out(X, name, scale=scale)
    loco_results.append({"name": name, "ec1_acc": ec1, "n_eval": n})
    if "REAL ESMC" in name:
        best_conf, best_name = conf, name

# ---- save ----
print("[3/3] Saving ...", flush=True)
with open(f"{OUT_DIR}/benchmark_esmc_sae.json", "w") as f:
    json.dump({
        "overall": results,
        "leave_one_class_out": loco_results,
        "bin_labels": bin_labels,
        "random_ec1_baseline": 1 / n_ec1,
        "n_proteins": n_proteins,
        "n_classes": n_classes,
    }, f, indent=2)

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel A: overall
names = [r["name"] for r in results]
top1  = [r["top1"]  for r in results]
top5  = [r["top5"]  for r in results]
auroc = [r["auroc"] for r in results]
xpos = np.arange(len(names))
short_names = [n.replace("REAL ESMC-SAE ", "ESMC-SAE\n").replace("Simulated SAE", "Sim SAE").replace("ESM-2 8M dense", "ESM-2 8M").replace(" (", "\n(").replace("3-mer binary", "3-mer") for n in names]
axes[0].bar(xpos - 0.25, top1,  width=0.25, label="Top-1", color="#3b82f6")
axes[0].bar(xpos,         top5,  width=0.25, label="Top-5", color="#10b981")
axes[0].bar(xpos + 0.25, auroc, width=0.25, label="AUROC", color="#f59e0b")
axes[0].set_xticks(xpos)
axes[0].set_xticklabels(short_names, rotation=0, ha="center", fontsize=8)
axes[0].set_ylabel("score")
axes[0].set_title(f"EC prediction — all Linear Probe ({n_classes} classes, {len(y_tr)} train / {len(y_te)} test)")
axes[0].legend(loc="lower right", fontsize=8)
axes[0].set_ylim(0, 1.05)
axes[0].grid(axis="y", alpha=0.3)

# Panel B: top-5 vs sequence similarity
colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
markers = ["o", "s", "^", "D", "P"]
for i, r in enumerate(results):
    vals = [b["top5"] if not np.isnan(b.get("top5", float('nan'))) else np.nan for b in r["per_bin"]]
    axes[1].plot(range(len(bin_labels)), vals, marker=markers[i % len(markers)],
                 color=colors[i % len(colors)], label=r["name"][:40], linewidth=2, markersize=7)
axes[1].set_xticks(range(len(bin_labels)))
axes[1].set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=9)
axes[1].set_xlabel("max 3-mer Jaccard to training set")
axes[1].set_ylabel("Top-5 accuracy")
axes[1].set_title("Accuracy vs sequence similarity (dark-matter bins on left)")
axes[1].legend(loc="lower right", fontsize=6)
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0, 1.05)

plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/benchmark_esmc_sae.png", dpi=140, bbox_inches="tight")
print(f"  saved {PLOT_DIR}/benchmark_esmc_sae.png", flush=True)

# ---- EC1 confusion matrix plot ----
if best_conf is not None:
    ec1_conf_norm = best_conf / (best_conf.sum(axis=1, keepdims=True) + 1e-9)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    im = ax2.imshow(ec1_conf_norm, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xticks(range(n_ec1)); ax2.set_xticklabels(ec1_le.classes_)
    ax2.set_yticks(range(n_ec1)); ax2.set_yticklabels(ec1_le.classes_)
    ax2.set_xlabel("predicted EC1"); ax2.set_ylabel("true EC1 (held-out)")
    ax2.set_title(f"EC1 confusion — {best_name} (leave-class-out)")
    for i in range(n_ec1):
        for j in range(n_ec1):
            ax2.text(j, i, f"{ec1_conf_norm[i,j]:.2f}", ha="center", va="center",
                     color="white" if ec1_conf_norm[i,j] > 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/ec1_confusion_esmc_sae.png", dpi=140, bbox_inches="tight")
    print(f"  saved {PLOT_DIR}/ec1_confusion_esmc_sae.png", flush=True)

# ---- comparison table ----
print("\n" + "=" * 85)
print("FEATURE COMPARISON — ALL LINEAR PROBE")
print("=" * 85)
print(f"{'Feature':38s}  {'Top-1':>6s}  {'Top-5':>6s}  {'AUROC':>6s}  {'EC1-LOCO':>9s}  {'Time':>6s}")
print("-" * 85)
for r in results:
    loco_match = [l for l in loco_results if l["name"] == r["name"]]
    loco_str = f"{loco_match[0]['ec1_acc']:.3f}" if loco_match else "  --"
    print(f"{r['name']:38s}  {r['top1']:6.3f}  {r['top5']:6.3f}  {r['auroc']:6.3f}  {loco_str:>9s}  {r['train_time']:5.1f}s")

print(f"\nRandom EC1 baseline: {1/n_ec1:.3f}")
print(f"Saved to {OUT_DIR}/benchmark_esmc_sae.json")
