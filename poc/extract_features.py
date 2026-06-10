"""
PoC: SAE-like binary features vs. ESM-2 dense vs. k-mer for EC number prediction.

Goal: show that *some* representation can predict EC for proteins with low
sequence identity to the training set, motivating the dark-matter use case.
"""
import os
import sys
import time
import json
import pickle
import warnings
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

DATA_TSV = "data/swissprot_enzymes.tsv"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load and clean data
# ---------------------------------------------------------------------------
print("[1/6] loading + cleaning SwissProt enzymes ...", flush=True)
df = pd.read_csv(DATA_TSV, sep="\t")
df = df.dropna(subset=["EC number", "Sequence"])
df = df[~df["EC number"].str.contains("-")]               # 3.4.24.-  ->  drop
df = df[df["EC number"].str.count(r"\.") == 3]            # only 4-level
df = df[df["Sequence"].str.len().between(120, 700)]
df = df.drop_duplicates(subset=["EC number", "Sequence"])
print(f"  filtered clean: {len(df)} entries", flush=True)

# Top-K most frequent EC classes (downstream multi-class)
K = 50
top_ec = df["EC number"].value_counts().head(K).index.tolist()
df = df[df["EC number"].isin(top_ec)].reset_index(drop=True)
print(f"  keep top-{K} EC classes: {len(df)} entries", flush=True)

# EC first digit (oxidoreductase / transferase / hydrolase / lyase / isomerase / ligase)
df["ec1"] = df["EC number"].str.split(".").str[0]
print(f"  ec1 distribution: {df['ec1'].value_counts().to_dict()}", flush=True)

# ---------------------------------------------------------------------------
# 2. k-mer Jaccard for sequence-similarity proxy
# ---------------------------------------------------------------------------
print("[2/6] computing 3-mer Jaccard (sequence identity proxy) ...", flush=True)
AA = "ACDEFGHIKLMNPQRSTVWY"
KMERS = [a+b+c for a in AA for b in AA for c in AA]   # 8000-dim
kmer_index = {k: i for i, k in enumerate(KMERS)}

def seq_to_kmer_set(seq, k=3):
    return set(seq[i:i+k] for i in range(len(seq)-k+1))

def seq_to_kmer_vec(seq):
    v = np.zeros(len(KMERS), dtype=np.uint8)
    for i in range(len(seq)-2):
        kmer = seq[i:i+3]
        if kmer in kmer_index:
            v[kmer_index[kmer]] = 1
    return v

t0 = time.time()
seqs = df["Sequence"].tolist()
accs = df["Entry"].tolist()
ecs = df["EC number"].tolist()

kmer_vecs = np.stack([seq_to_kmer_vec(s) for s in seqs])
print(f"  kmer matrix: {kmer_vecs.shape}, time {time.time()-t0:.1f}s", flush=True)

# pairwise Jaccard in chunks to avoid 5000x5000 blow-up
def jaccard_chunked(M, batch=500):
    n = M.shape[0]
    R = np.zeros((n, n), dtype=np.float32)
    Mb = M.astype(np.float32)
    for i in range(0, n, batch):
        Ai = Mb[i:i+batch]
        inter = Ai @ Mb.T
        a_sum = Ai.sum(1, keepdims=True)
        b_sum = Mb.sum(1, keepdims=True).T
        union = a_sum + b_sum - inter + 1e-6
        R[i:i+batch] = (inter / union).astype(np.float32)
    return R

t0 = time.time()
J = jaccard_chunked(kmer_vecs)
np.save(f"{OUT_DIR}/jaccard.npy", J)
print(f"  jaccard matrix: {J.shape}, time {time.time()-t0:.1f}s", flush=True)

# ---------------------------------------------------------------------------
# 3. ESM-2 8M embeddings (fast PoC)  -- substitute for ESMC+SAE
# ---------------------------------------------------------------------------
print("[3/6] computing ESM-2 8M mean-pooled embeddings (MPS) ...", flush=True)
import esm
model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
batch_converter = alphabet.get_batch_converter()
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device).eval()
print(f"  device: {device}, params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M",
      flush=True)

def embed_batch(seq_list, max_len=700, batch_size=16):
    embs = []
    for i in range(0, len(seq_list), batch_size):
        batch = seq_list[i:i+batch_size]
        batch = [(f"p{j}", s[:max_len]) for j, s in enumerate(batch)]
        _, _, tokens = batch_converter(batch)
        tokens = tokens.to(device)
        with torch.no_grad():
            out = model(tokens, repr_layers=[6], return_contacts=False)
        rep = out["representations"][6]                     # (B, L, 320)
        # mean-pool over residues (exclude special tokens positions 0 and -1)
        rep = rep[:, 1:-1, :].mean(dim=1)                  # (B, 320)
        embs.append(rep.cpu().numpy())
        if (i // batch_size) % 20 == 0:
            print(f"    {i+len(batch)}/{len(seq_list)}", flush=True)
    return np.concatenate(embs, axis=0)

t0 = time.time()
# truncate for speed
seqs_trunc = [s[:600] for s in seqs]
emb_dense = embed_batch(seqs_trunc, max_len=600, batch_size=32)
print(f"  dense embeddings: {emb_dense.shape}, time {time.time()-t0:.1f}s", flush=True)
np.save(f"{OUT_DIR}/esm2_8M_dense.npy", emb_dense)

# ---------------------------------------------------------------------------
# 4. SAE-like sparse binary features  (poor-man's SAE from ESM-2)
# ---------------------------------------------------------------------------
# Strategy: per-protein, take the top-K most-activated dimensions of ESM-2 dense
# embedding, set them to 1, rest 0.  K=64 mirrors the SAE TopK=64 of the paper.
print("[4/6] building SAE-like binary features (TopK=64 threshold) ...", flush=True)
K_TOP = 64
# rank per-protein: dims sorted descending; mark top-K
topk_idx = np.argpartition(-emb_dense, K_TOP, axis=1)[:, :K_TOP]
sae_bin = np.zeros_like(emb_dense, dtype=np.uint8)
rows = np.arange(emb_dense.shape[0])[:, None]
sae_bin[rows, topk_idx] = 1
print(f"  SAE-like binary: {sae_bin.shape}, density {sae_bin.mean():.4f}", flush=True)
np.save(f"{OUT_DIR}/sae_like_binary.npy", sae_bin)

# Also save the full kmer vec
np.save(f"{OUT_DIR}/kmer3.npy", kmer_vecs)
with open(f"{OUT_DIR}/meta.pkl", "wb") as f:
    pickle.dump({"accs": accs, "ecs": ecs, "ec1": df["ec1"].tolist()}, f)

print("\nALL FEATURES SAVED", flush=True)
print("  results/jaccard.npy")
print("  results/esm2_8M_dense.npy")
print("  results/sae_like_binary.npy")
print("  results/kmer3.npy")
print("  results/meta.pkl")
