"""
Leave-one-EC-class-out:  truly novel functional category.
Use LINEAR PROBE (Logistic Regression) for speed -- standard for evaluating
pre-trained feature quality.
"""
import os, json, time, pickle, warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, top_k_accuracy_score
warnings.filterwarnings("ignore")

OUT_DIR = "results"
emb_dense = np.load(f"{OUT_DIR}/esm2_8M_dense.npy")
sae_bin   = np.load(f"{OUT_DIR}/sae_like_binary.npy")
kmer3     = np.load(f"{OUT_DIR}/kmer3.npy")
with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
ecs  = meta["ecs"]; ec1s = meta["ec1"]
le = LabelEncoder(); y = le.fit_transform(ecs)
ec1_le = LabelEncoder(); y_ec1 = ec1_le.fit_transform(ec1s)
n_classes = len(le.classes_)
n_ec1 = len(ec1_le.classes_)
print(f"leave-one-EC-class-out: n={len(y)}  n_classes={n_classes}  n_ec1={n_ec1}", flush=True)

def evaluate(X, name, scale=False):
    t0 = time.time()
    n_eval = 0
    exact_acc_sum = 0.0
    ec1_acc_sum = 0.0
    ec1_conf = np.zeros((n_ec1, n_ec1), dtype=int)
    n_test_total = 0
    for c in range(n_classes):
        mask_te = y == c
        idx_te = np.where(mask_te)[0]
        idx_tr = np.where(~mask_te)[0]
        if len(idx_te) < 3 or len(idx_tr) < 100:
            continue
        Xtr = X[idx_tr]
        Xte = X[idx_te]
        if scale:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
        clf = LogisticRegression(max_iter=300, n_jobs=-1, C=1.0)
        clf.fit(Xtr, y[idx_tr])
        proba = clf.predict_proba(Xte)         # only over training classes
        # map to global
        global_proba = np.full((len(idx_te), n_classes), 1e-9, dtype=np.float32)
        for j, cls in enumerate(clf.classes_):
            global_proba[:, cls] = proba[:, j]
        pred = global_proba.argmax(1)
        # EC1 of predicted
        pred_ec1 = ec1_le.transform([le.classes_[p].split(".")[0] for p in pred])
        true_ec1 = y_ec1[idx_te]
        for t, p in zip(true_ec1, pred_ec1):
            ec1_conf[t, p] += 1
        n_eval += 1
        exact_acc_sum += (pred == c).sum() / len(idx_te)
        ec1_acc_sum += (pred_ec1 == true_ec1).sum() / len(idx_te)
        n_test_total += len(idx_te)
    exact = exact_acc_sum / n_eval
    ec1   = ec1_acc_sum / n_eval
    print(f"  {name:25s}  EC1={ec1:.3f}  exact={exact:.3f}  "
          f"(eval_classes={n_eval}/{n_classes}, time={time.time()-t0:.1f}s)", flush=True)
    return ec1, exact, n_eval, ec1_conf

results_loco = []
for X, name, scale in [
    (emb_dense, "ESM-2 8M dense",       True),
    (sae_bin,   "SAE-like TopK=64 bin", False),
    (kmer3,     "3-mer binary",         False),
]:
    ec1, exact, n_eval, ec1_conf = evaluate(X, name, scale=scale)
    results_loco.append({"name": name, "ec1_acc": ec1, "exact_acc": exact,
                          "n_eval": n_eval})
    # save per-method confusion
    np.save(f"{OUT_DIR}/loco_confusion_{name.replace(' ', '_').replace('=', '').replace('-', '')}.npy",
            ec1_conf)

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
ax.bar(xpos - 0.2, ec1_acc, width=0.4,
       label="EC1 super-class acc\n(can we tell it's a transferase / hydrolase / ...)",
       color="#10b981")
ax.bar(xpos + 0.2, exact_acc, width=0.4,
       label="Exact EC-4 acc\n(prediction lands on the held-out class)",
       color="#94a3b8")
ax.set_xticks(xpos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylabel("accuracy")
ax.set_title(f"Leave-one-EC-class-out ({n_classes} classes, linear probe)")
ax.set_ylim(0, 1.0)
ax.axhline(1/n_ec1, ls="--", color="gray", alpha=0.5,
           label=f"random baseline (1/{n_ec1} = {1/n_ec1:.2f})")
ax.legend(loc="upper right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/../plots/leave_class_out.png", dpi=140, bbox_inches="tight")
print(f"saved plots/leave_class_out.png", flush=True)

# confusion for ESM-2
ec1_conf_norm = ec1_conf / (ec1_conf.sum(axis=1, keepdims=True) + 1e-9)
fig2, ax2 = plt.subplots(figsize=(6, 5))
im = ax2.imshow(ec1_conf_norm, cmap="Blues", vmin=0, vmax=1)
ax2.set_xticks(range(n_ec1))
ax2.set_xticklabels(ec1_le.classes_)
ax2.set_yticks(range(n_ec1))
ax2.set_yticklabels(ec1_le.classes_)
ax2.set_xlabel("predicted EC1")
ax2.set_ylabel("true EC1 (held-out)")
ax2.set_title("EC1 confusion  (ESM-2 8M, leave-EC-class-out)")
for i in range(n_ec1):
    for j in range(n_ec1):
        ax2.text(j, i, f"{ec1_conf_norm[i,j]:.2f}", ha="center", va="center",
                 color="white" if ec1_conf_norm[i,j] > 0.5 else "black", fontsize=9)
plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/../plots/ec1_confusion.png", dpi=140, bbox_inches="tight")
print(f"saved plots/ec1_confusion.png", flush=True)
