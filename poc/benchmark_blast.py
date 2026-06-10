"""
BLASTp benchmark on the same 4,868 microbial enzyme dataset.
Uses the identical 80/20 split as our ESMC-SAE benchmark (seed=42).
BLASTp is the gold-standard homology-based method used daily by microbiologists.
"""
import os, pickle, time, subprocess, tempfile
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit

OUT_DIR = "results"
PREFIX = "microbial_esmc_sae"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- Load data ----
print("[1/4] Loading data...", flush=True)
with open(f"{OUT_DIR}/{PREFIX}_meta.pkl", "rb") as f:
    meta = pickle.load(f)

ecs = meta["ecs"]
ec3s = [".".join(e.split(".")[:3]) for e in ecs]
n = len(ecs)

# Filter to EC3 with >=5 members (same as benchmark)
from collections import Counter
ec3_counts = Counter(ec3s)
valid_ec3 = {e for e, c in ec3_counts.items() if c >= 5}
keep_idx = [i for i, e in enumerate(ec3s) if e in valid_ec3]
n_filtered = len(keep_idx)

ec3s_f = [ec3s[i] for i in keep_idx]
ecs_f = [ecs[i] for i in keep_idx]

# Load sequences
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

seqs_all = [acc2seq.get(meta["accs"][i], "") for i in range(n)]
seqs_f = [seqs_all[i] for i in keep_idx]
accs_f = [meta["accs"][i] for i in keep_idx]
assert len(seqs_f) == n_filtered

# ---- 80/20 split (same as benchmark, seed=42) ----
print("[2/4] Creating train/test split...", flush=True)
le = LabelEncoder()
y = le.fit_transform(ec3s_f)

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(sss.split(np.zeros(n_filtered), y))
print(f"  Train: {len(tr)}, Test: {len(te)}", flush=True)

# ---- Create BLAST database from training sequences ----
print("[3/4] Building BLAST database and running queries...", flush=True)

tmpdir = tempfile.mkdtemp(prefix="blast_")
db_fasta = os.path.join(tmpdir, "train.fasta")
db_name = os.path.join(tmpdir, "train_db")

# Write training FASTA with EC3 in header
with open(db_fasta, "w") as f:
    for i in tr:
        acc = accs_f[i]
        ec3 = ec3s_f[i]
        seq = seqs_f[i]
        f.write(f">{acc}|EC={ec3}\n{seq}\n")

# makeblastdb
subprocess.run(["makeblastdb", "-in", db_fasta, "-dbtype", "prot", "-out", db_name,
                "-logfile", os.path.join(tmpdir, "makeblastdb.log")],
               check=True, capture_output=True)
print(f"  BLAST DB: {len(tr)} sequences", flush=True)

# Query each test sequence
query_fasta = os.path.join(tmpdir, "test.fasta")
blast_out = os.path.join(tmpdir, "blast_results.txt")

with open(query_fasta, "w") as f:
    for i in te:
        f.write(f">{accs_f[i]}|trueEC={ec3s_f[i]}\n{seqs_f[i]}\n")

# Run BLASTp (single-threaded for reproducibility, e-value 1e-3, top 5 hits)
subprocess.run(["blastp", "-query", query_fasta, "-db", db_name,
                "-out", blast_out, "-outfmt", "6 qseqid sseqid pident evalue bitscore",
                "-evalue", "1e-3", "-max_target_seqs", "5", "-num_threads", "4"],
               check=True, capture_output=True)
print(f"  BLASTp complete", flush=True)

# ---- Parse results ----
print("[4/4] Evaluating BLASTp accuracy...", flush=True)

# Parse blast output
blast_hits = {}
with open(blast_out) as f:
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 5:
            continue
        q_id = parts[0]
        s_id = parts[1]  # format: acc|EC=1.1.1
        pident = float(parts[2])
        evalue = float(parts[3])
        if "EC=" in s_id:
            pred_ec3 = s_id.split("EC=")[1]
        else:
            continue
        if q_id not in blast_hits:
            blast_hits[q_id] = []
        blast_hits[q_id].append((pred_ec3, pident, evalue))

# Evaluate
top1_correct = 0
top5_correct = 0
no_hits = 0
per_bin_correct = {b: 0 for b in range(6)}
per_bin_total = {b: 0 for b in range(6)}

# Compute Jaccard bins for stratification (same as benchmark)
from sklearn.feature_extraction.text import CountVectorizer

def seq_to_kmers(s, k=3):
    return " ".join(s[i:i+k] for i in range(len(s)-k+1))

kmer_vec = CountVectorizer(analyzer="char", ngram_range=(3,3), lowercase=False,
                           max_features=8000, binary=True)
kmer_all = kmer_vec.fit_transform(seqs_f).toarray().astype(np.uint8)
kmer_tr = kmer_all[tr]
kmer_te = kmer_all[te]

# Sample 2000 training for Jaccard
import random
random.seed(42)
max_comp = min(2000, len(tr))
sample_tr = random.sample(range(len(tr)), max_comp)
kmer_tr_s = kmer_tr[sample_tr]

max_sim = np.zeros(len(te))
for i in range(len(te)):
    a = kmer_te[i]
    inter = (a & kmer_tr_s).sum(axis=1)
    union = (a | kmer_tr_s).sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        jacc = np.where(union > 0, inter / union, 0)
    max_sim[i] = jacc.max()

bins = [0.0, 0.20, 0.30, 0.40, 0.50, 0.65, 1.01]
bin_labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.65", ">=0.65"]
bin_idx = np.digitize(max_sim, bins[1:-1])

for i, te_idx in enumerate(te):
    query_id = f"{accs_f[te_idx]}|trueEC={ec3s_f[te_idx]}"
    true_ec3 = ec3s_f[te_idx]
    bi = bin_idx[i]

    if query_id not in blast_hits or len(blast_hits[query_id]) == 0:
        no_hits += 1
        continue

    hits = blast_hits[query_id]
    # Top-1
    if hits[0][0] == true_ec3:
        top1_correct += 1

    # Top-5
    top5_ec3s = [h[0] for h in hits[:5]]
    if true_ec3 in top5_ec3s:
        top5_correct += 1

    # Per-bin (top-5)
    if true_ec3 in [h[0] for h in hits[:5]]:
        per_bin_correct[bi] += 1
    per_bin_total[bi] += 1

n_test = len(te)
top1 = top1_correct / n_test
top5 = top5_correct / n_test
no_hit_rate = no_hits / n_test

print(f"\n{'='*60}")
print(f"BLASTp RESULTS (80/20 split, {n_test} test proteins)")
print(f"{'='*60}")
print(f"  Top-1 accuracy:  {top1:.4f} ({top1_correct}/{n_test})")
print(f"  Top-5 accuracy:  {top5:.4f} ({top5_correct}/{n_test})")
print(f"  No hits found:   {no_hits}/{n_test} ({no_hit_rate:.3f})")
print(f"\n  Per-bin Top-5 accuracy (by sequence similarity to training):")
for b in range(6):
    if per_bin_total[b] > 0:
        print(f"    {bin_labels[b]:10s}: {per_bin_correct[b]/per_bin_total[b]:.4f}  (n={per_bin_total[b]})")
    else:
        print(f"    {bin_labels[b]:10s}: --  (n=0)")

# Print comparison with ESMC-SAE
print(f"\n{'='*60}")
print(f"COMPARISON: BLASTp vs ESMC-SAE")
print(f"{'='*60}")
print(f"{'Method':30s} {'Top-1':>7s} {'Top-5':>7s}")
print(f"{'-'*46}")
print(f"{'BLASTp (homology transfer)':30s} {top1:7.4f} {top5:7.4f}")
print(f"{'ESMC-SAE binary (16384d)':30s} {0.7885:7.4f} {0.8850:7.4f}")
print(f"{'ESMC-SAE combined (32768d)':30s} {0.8563:7.4f} {0.9045:7.4f}")

# Save results
import json
results = {
    "method": "BLASTp",
    "n_train": len(tr),
    "n_test": n_test,
    "top1": top1,
    "top5": top5,
    "no_hit_rate": no_hit_rate,
    "per_bin": [{"bin": bin_labels[b], "n": per_bin_total[b],
                  "top5": per_bin_correct[b]/per_bin_total[b] if per_bin_total[b] > 0 else None}
                 for b in range(6)],
    "esmc_sae_binary_top1": 0.7885,
    "esmc_sae_combined_top1": 0.8563,
}
with open(f"{OUT_DIR}/benchmark_blast.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to {OUT_DIR}/benchmark_blast.json")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
print("Done!")
