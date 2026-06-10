"""
Quick leave-one-EC3-class-out → EC1 accuracy.
Only trains 7-class EC1 classifier per held-out EC3 class (much faster than 161-class).
"""
import pickle, json, time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

OUT_DIR = "results"
PREFIX = "microbial_esmc_sae"

# Load
print("Loading...", flush=True)
weights = np.load(f"{OUT_DIR}/{PREFIX}_weights.npy")
binary = np.load(f"{OUT_DIR}/{PREFIX}_binary.npy")
with open(f"{OUT_DIR}/{PREFIX}_meta.pkl", "rb") as f:
    meta = pickle.load(f)

ec3s = [".".join(e.split(".")[:3]) for e in meta["ecs"]]
ec1s = meta["ec1s"]

# Filter EC3 classes with >= 5 members
from collections import Counter
ec3_counts = Counter(ec3s)
valid_ec3 = {e for e, c in ec3_counts.items() if c >= 5}
keep = [i for i, e in enumerate(ec3s) if e in valid_ec3]

X_bin = binary[keep]
X_wgt = weights[keep]
ec3_f = [ec3s[i] for i in keep]
ec1_f = [ec1s[i] for i in keep]

# Encode EC1 (target) and EC3 (for hold-out)
le_ec1 = LabelEncoder(); y_ec1 = le_ec1.fit_transform(ec1_f)
le_ec3 = LabelEncoder(); y_ec3 = le_ec3.fit_transform(ec3_f)
n_ec1 = len(le_ec1.classes_)
n_ec3 = len(le_ec3.classes_)
n = len(keep)

print(f"Proteins: {n}, EC3 classes: {n_ec3}, EC1 classes: {n_ec1}", flush=True)

# Select top-60 most populated EC3 classes
class_sizes = [(c, (y_ec3 == c).sum()) for c in range(n_ec3)]
class_sizes.sort(key=lambda x: -x[1])
eval_classes = [c for c, sz in class_sizes[:60] if sz >= 5]
print(f"Evaluating {len(eval_classes)} classes", flush=True)

# ---- EC1 leave-one-EC3-class-out ----
def loco(X, name, scale_x=False):
    from sklearn.preprocessing import StandardScaler
    t0 = time.time()
    n_eval = 0
    correct = 0
    total = 0

    for c in eval_classes:
        te_idx = np.where(y_ec3 == c)[0]
        tr_idx = np.where(y_ec3 != c)[0]
        if len(te_idx) < 3 or len(tr_idx) < 200:
            continue

        Xtr, Xte = X[tr_idx], X[te_idx]
        ytr, yte = y_ec1[tr_idx], y_ec1[te_idx]

        if scale_x:
            sc = StandardScaler().fit(Xtr)
            Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)

        clf = LogisticRegression(max_iter=200, n_jobs=-1, C=1.0)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        correct += (pred == yte).sum()
        total += len(yte)
        n_eval += 1

        if n_eval % 15 == 0:
            print(f"  {n_eval}/{len(eval_classes)}: EC1={correct/total:.4f}", flush=True)

    acc = correct / total
    print(f"  {name:35s} EC1={acc:.4f} ({n_eval} classes, {time.time()-t0:.1f}s)", flush=True)
    return acc

print("\n--- Leave-one-EC3-class-out → EC1 ---")
print(f"  Random baseline: {1/n_ec1:.4f}\n")

results = []
for X, name, scale in [
    (X_bin, "ESMC-SAE binary (16384d)", False),
    (X_wgt, "ESMC-SAE weights (16384d)", True),
]:
    acc = loco(X, name, scale_x=scale)
    results.append({"name": name, "ec1_loco": acc})

# Compare with k-mer baseline (use CountVectorizer)
from sklearn.feature_extraction.text import CountVectorizer
seqs = []
with open("data/microbial_enzymes_5k.tsv") as f:
    header = f.readline().strip().split("\t")
    a_idx = header.index("Entry")
    s_idx = header.index("Sequence")
    acc2seq = {}
    for line in f:
        row = line.strip().split("\t")
        if len(row) > s_idx:
            acc2seq[row[a_idx]] = row[s_idx]
seqs_f = [acc2seq.get(meta["accs"][i], "") for i in keep]
assert len(seqs_f) == n

vec = CountVectorizer(analyzer="char", ngram_range=(3,3), lowercase=False, max_features=8000, binary=True)
X_kmer = vec.fit_transform(seqs_f).toarray().astype(np.uint8)

acc_kmer = loco(X_kmer, "3-mer binary (8000d)", scale_x=False)
results.insert(0, {"name": "3-mer binary (8000d)", "ec1_loco": acc_kmer})

# Save
output = {
    "evaluation": "leave-one-EC3-class-out → EC1",
    "n_proteins": n,
    "n_ec3_classes_evaluated": len(eval_classes),
    "random_baseline": 1/n_ec1,
    "results": results,
}
with open(f"{OUT_DIR}/loco_quick_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n{'='*70}")
print("LEAVE-ONE-CLASS-OUT EC1 ACCURACY")
print(f"{'Feature':40s} {'EC1 acc':>8s}  vs random")
print(f"{'-'*70}")
for r in results:
    print(f"{r['name']:40s} {r['ec1_loco']:8.4f}  {r['ec1_loco']/(1/n_ec1):.1f}x")
print(f"\nRandom: {1/n_ec1:.4f}")
print(f"Saved to {OUT_DIR}/loco_quick_results.json")
