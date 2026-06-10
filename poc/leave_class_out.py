"""
Leave-one-EC-class-out evaluation:  truly novel functional category.

For each EC class (e.g. EC 2.7.1.1), hold it out completely from training.
Train on the remaining 49 classes.  Test on the held-out class.
Metric: macro-F1 over the 50 classes (one-vs-rest) + the held-out's
       predicted super-class (EC1) accuracy.

This is the "dark matter enzyme" scenario:  we encounter a new enzyme
in metagenomes whose exact function has never been curated -- can we
at least tell it's a hydrolase / transferase / ...?
"""
import os
import json
import time
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, top_k_accuracy_score
import warnings
warnings.filterwarnings("ignore")

OUT_DIR = "results"
PLOT_DIR = "plots"
emb_dense = np.load(f"{OUT_DIR}/esm2_8M_dense.npy")
sae_bin   = np.load(f"{OUT_DIR}/sae_like_binary.npy")
kmer3     = np.load(f"{OUT_DIR}/kmer3.npy")
with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
ecs  = meta["ecs"]
ec1s = meta["ec1"]
le = LabelEncoder()
y = le.fit_transform(ecs)
n_classes = len(le.classes_)
ec1_le = LabelEncoder()
y_ec1 = ec1_le.fit_transform(ec1s)

print(f"leave-one-EC-class-out, n_proteins={len(y)}, n_classes={n_classes}, n_ec1={len(ec1_le.classes_)}", flush=True)

def evaluate(X, name):
    t0 = time.time()
    pred_top1 = np.zeros(len(y), dtype=int)
    pred_proba = np.zeros((len(y), n_classes), dtype=np.float32)
    held_correct = []
    held_pred_ec1_correct = []
    held_class_acc = []
    for c in range(n_classes):
        mask = y == c
        idx_held = np.where(mask)[0]
        idx_train = np.where(~mask)[0]
        if len(idx_held) < 3 or len(idx_train) < 100:
            continue
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05,
                                              max_depth=6, random_state=42)
        clf.fit(X[idx_train], y[idx_train])
        proba = clf.predict_proba(X[idx_held])
        # align proba columns to global class ids
        global_proba = np.zeros((len(idx_held), n_classes), dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            global_proba[:, cls] = proba[:, j]
        pred = global_proba.argmax(1)
        pred_top1[idx_held] = pred
        pred_proba[idx_held] = global_proba
        # per-class:  predicted EC1 of the held-out class
        true_ec1 = y_ec1[idx_held]
        pred_ec1_of_pred = ec1_le.transform([le.classes_[p].split(".")[0] for p in pred])
        held_pred_ec1_correct.append((pred_ec1_of_pred == true_ec1).mean())
        held_class_acc.append((pred == c).mean())
    # overall:  when a protein is held out, can we tell its EC1?
    overall_ec1_acc = np.mean(held_pred_ec1_correct)
    overall_class_acc = np.mean(held_class_acc)
    n_eval_classes = len(held_class_acc)
    print(f"  {name:25s}  EC1 super-class acc={overall_ec1_acc:.3f}  "
          f"exact EC acc={overall_class_acc:.3f}  "
          f"(n_eval_classes={n_eval_classes}, time={time.time()-t0:.1f}s)", flush=True)
    return overall_ec1_acc, overall_class_acc, n_eval_classes

results_loco = []
for X, name in [(emb_dense, "ESM-2 8M dense"),
                (sae_bin,   "SAE-like TopK=64"),
                (kmer3,     "3-mer binary")]:
    r = evaluate(X, name)
    results_loco.append({"name": name, **dict(zip(["ec1_acc", "exact_acc", "n_eval"], r))})

with open(f"{OUT_DIR}/loco_results.json", "w") as f:
    json.dump(results_loco, f, indent=2)

# ---- plot ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(7, 4.2))
names = [r["name"] for r in results_loco]
ec1_acc = [r["ec1_acc"] for r in results_loco]
exact_acc = [r["exact_acc"] for r in results_loco]
xpos = np.arange(len(names))
ax.bar(xpos - 0.2, ec1_acc, width=0.4, label="EC1 super-class accuracy\n(can we tell it's a transferase / hydrolase / ...)", color="#10b981")
ax.bar(xpos + 0.2, exact_acc, width=0.4, label="Exact EC-4 prediction\n(held-out class as test)", color="#94a3b8")
ax.set_xticks(xpos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("accuracy")
ax.set_title("Leave-one-EC-class-out  (truly novel functional category)")
ax.set_ylim(0, 1.0)
ax.axhline(1/len(ec1_le.classes_), ls="--", color="gray", alpha=0.5,
           label=f"random baseline (1/{len(ec1_le.classes_)} = {1/len(ec1_le.classes_):.2f})")
ax.legend(loc="upper right", fontsize=8.5)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/leave_class_out.png", dpi=140, bbox_inches="tight")
print(f"\nsaved {PLOT_DIR}/leave_class_out.png", flush=True)

# cross-tab of held-out -> predicted EC1
print("\nEC1 confusion insight: for each held-out class, where did it get predicted?")
ec1_confusion = np.zeros((len(ec1_le.classes_), len(ec1_le.classes_)), dtype=int)
for c in range(n_classes):
    mask = y == c
    idx_held = np.where(mask)[0]
    idx_train = np.where(~mask)[0]
    if len(idx_held) < 3 or len(idx_train) < 100:
        continue
    X = emb_dense
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=6, random_state=42)
    clf.fit(X[idx_train], y[idx_train])
    proba = clf.predict_proba(X[idx_held])
    global_proba = np.zeros((len(idx_held), n_classes), dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        global_proba[:, cls] = proba[:, j]
    pred = global_proba.argmax(1)
    true_ec1 = y_ec1[idx_held]
    pred_ec1 = ec1_le.transform([le.classes_[p].split(".")[0] for p in pred])
    for t, p in zip(true_ec1, pred_ec1):
        ec1_confusion[t, p] += 1

# normalize per row
ec1_confusion_norm = ec1_confusion / (ec1_confusion.sum(axis=1, keepdims=True) + 1e-9)
print("\nEC1 confusion (rows=true, cols=predicted, normalized):")
print("              " + "  ".join(f"{ec1_le.classes_[i]:>4s}" for i in range(len(ec1_le.classes_))))
for i in range(len(ec1_le.classes_)):
    print(f"  {ec1_le.classes_[i]:>4s}        " + "  ".join(f"{ec1_confusion_norm[i,j]:.2f}" for j in range(len(ec1_le.classes_))))

fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(ec1_confusion_norm, cmap="Blues", vmin=0, vmax=1)
ax2.set_xticks(range(len(ec1_le.classes_)))
ax2.set_xticklabels(ec1_le.classes_)
ax2.set_yticks(range(len(ec1_le.classes_)))
ax2.set_yticklabels(ec1_le.classes_)
ax2.set_xlabel("predicted EC1")
ax2.set_ylabel("true EC1 (held-out)")
ax2.set_title("EC1 super-class confusion (ESM-2 8M, leave-EC-class-out)")
for i in range(len(ec1_le.classes_)):
    for j in range(len(ec1_le.classes_)):
        ax2.text(j, i, f"{ec1_confusion_norm[i,j]:.2f}", ha="center", va="center",
                 color="white" if ec1_confusion_norm[i,j] > 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{PLOT_DIR}/ec1_confusion.png", dpi=140, bbox_inches="tight")
print(f"saved {PLOT_DIR}/ec1_confusion.png", flush=True)
