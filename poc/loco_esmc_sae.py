"""Quick leave-one-EC-class-out for ESMC-SAE features only."""
import numpy as np, pickle, time
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUT_DIR = "results"

print("Loading...", flush=True)
esmc_b = np.load(f"{OUT_DIR}/esmc_sae_binary.npy")
esmc_w = np.load(f"{OUT_DIR}/esmc_sae_weights.npy")
with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
ecs = meta["ecs"]; ec1s = meta["ec1"]
le = LabelEncoder(); y = le.fit_transform(ecs)
n_classes = len(le.classes_)
ec1_le = LabelEncoder(); y_ec1 = ec1_le.fit_transform(ec1s)
n_ec1 = len(ec1_le.classes_)

print(f"n={len(y)} classes={n_classes} ec1={n_ec1} random_baseline={1/n_ec1:.3f}", flush=True)

def loco(X, name, scale=False):
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
        clf = LogisticRegression(max_iter=200, C=1.0, solver='liblinear', penalty='l1')
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
        if (c + 1) % 10 == 0:
            print(f"  {name:35s}  [{c+1:2d}/{n_classes}] EC1={ec1_acc_sum/n_eval:.3f}", flush=True)
    ec1_acc = ec1_acc_sum / n_eval
    print(f"  {name:35s}  EC1={ec1_acc:.3f}  ({n_eval}/{n_classes} classes, {time.time()-t0:.1f}s)", flush=True)
    return ec1_acc, ec1_conf

print("\nLeave-one-EC-class-out — ESMC-SAE only", flush=True)
ec1_b, conf_b = loco(esmc_b, "REAL ESMC-SAE binary (16384d)")
ec1_w, conf_w = loco(esmc_w, "REAL ESMC-SAE weights (16384d)", scale=True)

# Save
import json
results = [
    {"name": "REAL ESMC-SAE binary (16384d)", "ec1_acc": ec1_b},
    {"name": "REAL ESMC-SAE weights (16384d)", "ec1_acc": ec1_w},
]
with open(f"{OUT_DIR}/loco_esmc_sae.json", "w") as f:
    json.dump({"results": results, "random_baseline": 1/n_ec1}, f, indent=2)

np.save(f"{OUT_DIR}/loco_conf_esmc_sae_binary.npy", conf_b)
np.save(f"{OUT_DIR}/loco_conf_esmc_sae_weights.npy", conf_w)

print(f"\nSaved. Binary EC1={ec1_b:.3f}, Weights EC1={ec1_w:.3f}", flush=True)
