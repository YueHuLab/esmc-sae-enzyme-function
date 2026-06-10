"""
Extract ESMC-6B + SAE features for 11,392 microbial SwissProt enzymes.
Saves features + metadata for the expanded benchmark.
"""
import os, pickle, time
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
SAE_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B-sae-layer60-k64-codebook16384"
DATA_FILE = "data/microbial_enzymes_5k.tsv"
OUT_DIR = "results"
BATCH_SIZE = 8
TOP_K = 64
FEATURE_DIM = 16384
SAE_LAYER = 60

OUT_PREFIX = "microbial_esmc_sae"
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 1. load sequences and metadata ----
print("[1/4] Loading sequences ...", flush=True)
seqs, accs, ecs, organism_names, lineages = [], [], [], [], []

with open(DATA_FILE) as f:
    header = f.readline().strip().split("\t")
    acc_idx = header.index("Entry")
    seq_idx = header.index("Sequence")
    ec_idx = header.index("EC number")
    org_idx = header.index("Organism") if "Organism" in header else header.index("Organism (ID)")
    lin_idx = header.index("Lineage") if "Lineage" in header else None

    for line in f:
        row = line.strip().split("\t")
        if len(row) > max(acc_idx, seq_idx, ec_idx):
            seqs.append(row[seq_idx])
            accs.append(row[acc_idx])
            ecs.append(row[ec_idx])
            organism_names.append(row[org_idx] if len(row) > org_idx else "")
            lineages.append(row[lin_idx] if lin_idx and len(row) > lin_idx else "")

n_proteins = len(seqs)
print(f"  Loaded {n_proteins} sequences", flush=True)

# Parse EC1 classes
ec1s = [ec.split(".")[0] for ec in ecs]
ec1_set = sorted(set(ec1s))
print(f"  EC1 classes: {ec1_set}", flush=True)

# ---- 2. load models ----
print("[2/4] Loading models (ESMC-6B on MPS, SAE on CPU) ...", flush=True)
t0 = time.time()

tokenizer = AutoTokenizer.from_pretrained(ESMC_PATH)
sae = AutoModel.from_pretrained(SAE_PATH, torch_dtype=torch.bfloat16, device="cpu")
sae.initialize_layers([SAE_LAYER])
sae_layer = sae.layers[str(SAE_LAYER)]
sae_layer.eval()
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
        output = model(**inputs, output_hidden_states=True)
        h60 = output.hidden_states[SAE_LAYER]
        h60_cpu = h60.to("cpu").float()
        sae_out = sae_layer(h60_cpu)
        fm = sae_out.feature_magnitudes

        for j in range(fm.shape[0]):
            protein_vec = fm[j].mean(dim=0).detach().numpy()
            all_weights[i + j] = protein_vec

    elapsed = time.time() - t0
    done = min(i + BATCH_SIZE, n_proteins)
    rate = done / elapsed if elapsed > 0 else 0
    eta = (n_proteins - done) / rate if rate > 0 else 0
    if i % 40 == 0 or done >= n_proteins:
        print(f"  {done:5d}/{n_proteins}  ({done/n_proteins*100:.1f}%)  "
              f"ETA {eta/60:.1f} min  ({rate:.1f} prot/s)", flush=True)

print(f"  Extraction done in {time.time()-t0:.1f}s", flush=True)

# ---- 4. binarize & save ----
print("[4/4] Binarizing & saving ...", flush=True)

binary = np.zeros_like(all_weights, dtype=np.uint8)
for i in range(n_proteins):
    top_idx = np.argpartition(-all_weights[i], TOP_K)[:TOP_K]
    binary[i, top_idx] = 1

print(f"  Active features per protein: {binary.sum(1).mean():.1f}", flush=True)

np.save(f"{OUT_DIR}/{OUT_PREFIX}_weights.npy", all_weights)
np.save(f"{OUT_DIR}/{OUT_PREFIX}_binary.npy", binary)

# Save metadata
meta = {
    "n_proteins": n_proteins,
    "accs": accs,
    "ecs": ecs,
    "ec1s": ec1s,
    "organism_names": organism_names,
    "lineages": lineages,
    "feature_dim": FEATURE_DIM,
    "top_k": TOP_K,
    "sae_layer": SAE_LAYER,
    "codebook_size": FEATURE_DIM,
}
with open(f"{OUT_DIR}/{OUT_PREFIX}_meta.pkl", "wb") as f:
    pickle.dump(meta, f)

print(f"  Saved {OUT_PREFIX}_weights.npy  ({all_weights.nbytes/1e6:.1f} MB)", flush=True)
print(f"  Saved {OUT_PREFIX}_binary.npy   ({binary.nbytes/1e6:.1f} MB)", flush=True)
print(f"  Saved {OUT_PREFIX}_meta.pkl", flush=True)
print("Done!", flush=True)
