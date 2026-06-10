"""
Extract real ESMC-6B + SAE (layer 60, K=64, codebook 16384) features
for the PoC SwissProt enzymes.

Hybrid approach: ESMC-6B on MPS (fast transformer), SAE on CPU (avoids MPS sparse bug).

Produces:
  - esmc_sae_weights.npy: N × 16384 float32 (per-protein, mean-pool over residues)
  - esmc_sae_binary.npy:  N × 16384 uint8 (top-64 features per protein)
"""
import os, pickle, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
SAE_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B-sae-layer60-k64-codebook16384"
DATA_DIR = "data"
OUT_DIR = "results"
BATCH_SIZE = 4
TOP_K = 64
FEATURE_DIM = 16384
SAE_LAYER = 60

os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. load sequences (only those used in PoC) ----
print("[1/4] Loading sequences ...", flush=True)
# Load the meta to get accession list used in PoC (after cleaning/filtering)
with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
poq_accs = set(meta["accs"])  # accessions used in PoC after cleaning

seqs = []
accs_loaded = []
with open(f"{DATA_DIR}/swissprot_enzymes.tsv") as f:
    header = f.readline().strip().split("\t")
    acc_idx = header.index("Entry")
    seq_idx = header.index("Sequence")
    for line in f:
        row = line.strip().split("\t")
        if len(row) > seq_idx and row[acc_idx] in poq_accs:
            seqs.append(row[seq_idx])
            accs_loaded.append(row[acc_idx])

n_proteins = len(seqs)
print(f"  Loaded {n_proteins} sequences (filtered to PoC accessions)", flush=True)

# ---- 2. load models ----
print("[2/4] Loading models (ESMC-6B on MPS, SAE on CPU) ...", flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(ESMC_PATH)

# SAE on CPU to avoid MPS sparse-tensor bug
sae = AutoModel.from_pretrained(SAE_PATH, torch_dtype=torch.bfloat16, device="cpu")
sae.initialize_layers([SAE_LAYER])
sae_layer = sae.layers[str(SAE_LAYER)]
sae_layer.eval()

# ESMC-6B on MPS for fast transformer inference
model = AutoModel.from_pretrained(ESMC_PATH, torch_dtype=torch.bfloat16, device_map="mps").eval()

print(f"  Models loaded in {time.time()-t0:.1f}s", flush=True)

# ---- 3. extract SAE features ----
print(f"[3/4] Extracting SAE features (batch_size={BATCH_SIZE}) ...", flush=True)

all_weights = np.zeros((n_proteins, FEATURE_DIM), dtype=np.float32)
t0 = time.time()

for i in range(0, n_proteins, BATCH_SIZE):
    batch_seqs = seqs[i : i + BATCH_SIZE]
    inputs = tokenizer(batch_seqs, return_tensors="pt", padding=True)
    inputs = {k: v.to("mps") for k, v in inputs.items()}

    with torch.inference_mode():
        # Get layer 60 hidden states from ESMC
        output = model(**inputs, output_hidden_states=True)
        h60 = output.hidden_states[SAE_LAYER]  # (batch, N_res, 2560) on MPS

        # Move to CPU for SAE
        h60_cpu = h60.to("cpu").float()

        # Run SAE on CPU (inside inference_mode to avoid grad tracking)
        sae_out = sae_layer(h60_cpu)  # ESMCSAEOutput
        fm = sae_out.feature_magnitudes  # (batch, N_res, 16384)

        for j in range(fm.shape[0]):
            # Mean-pool over residues → per-protein vector
            protein_vec = fm[j].mean(dim=0).detach().numpy()  # (16384,)
            all_weights[i + j] = protein_vec

    elapsed = time.time() - t0
    done = i + len(batch_seqs)
    rate = done / elapsed if elapsed > 0 else 0
    eta = (n_proteins - done) / rate if rate > 0 else 0
    if i % 20 == 0 or done >= n_proteins:
        print(f"  {done:5d}/{n_proteins}  ({done/n_proteins*100:.1f}%)  "
              f"ETA {eta/60:.1f} min  ({rate:.1f} prot/s)", flush=True)

print(f"  Extraction done in {time.time()-t0:.1f}s", flush=True)

# ---- 4. binarize & save ----
print("[4/4] Binarizing & saving ...", flush=True)

# Top-64 features per protein (matching SAE K=64)
binary = np.zeros_like(all_weights, dtype=np.uint8)
for i in range(n_proteins):
    top_idx = np.argpartition(-all_weights[i], TOP_K)[:TOP_K]
    binary[i, top_idx] = 1

active_per_prot = binary.sum(1).mean()
print(f"  binary: {binary.shape}, active features per protein: {active_per_prot:.1f}", flush=True)

np.save(f"{OUT_DIR}/esmc_sae_weights.npy", all_weights)
np.save(f"{OUT_DIR}/esmc_sae_binary.npy", binary)

# Update meta
with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
    meta = pickle.load(f)
meta["esmc_sae_weights"] = f"{OUT_DIR}/esmc_sae_weights.npy"
meta["esmc_sae_binary"] = f"{OUT_DIR}/esmc_sae_binary.npy"
with open(f"{OUT_DIR}/meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"  Saved esmc_sae_weights.npy ({all_weights.nbytes/1e6:.1f} MB)", flush=True)
print(f"  Saved esmc_sae_binary.npy  ({binary.nbytes/1e6:.1f} MB)", flush=True)
print("Done.", flush=True)
