"""
Dark matter enzyme discovery using ESMC-SAE features.
Selects dark clusters from representative_proteins.parquet,
fetches sequences via UniProt, predicts EC, validates against Pfam.

For the Nature Microbiology paper: demonstrates that ESMC-SAE can
predict function for proteins with no known EC annotation.
"""
import os, pickle, json, time, urllib.request, urllib.parse, re
import numpy as np
import pandas as pd
import torch
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

from transformers import AutoModel, AutoTokenizer

# ---- config ----
OUT_DIR = "results"
PLOT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

ESMC_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B"
SAE_PATH = "/Users/huyue/esm-c-fold2/ESMC-6B-sae-layer60-k64-codebook16384"
REP_PARQUET = "data/representative_proteins.parquet"
TRAIN_TSV = "data/microbial_enzymes_5k.tsv"

BATCH_SIZE = 4
FEATURE_DIM = 16384
TOP_K = 64
SAE_LAYER = 60

# ---- Step 1: Select dark candidates from representative_proteins ----
print("[1/6] Selecting dark matter candidates...", flush=True)

rep = pd.read_parquet(REP_PARQUET)

# Dark = 0% characterized, has Pfam, poorly named (tier 3-5)
# Focus on potential enzymes: look for Pfams with "hydrolase", "transferase", etc.
enzyme_pfam_keywords = [
    "hydrolase", "transferase", "oxidoreductase", "lyase", "isomerase", "ligase",
    "dehydrogenase", "kinase", "phosphatase", "protease", "peptidase", "esterase",
    "synthase", "synthetase", "decarboxylase", "aminotransferase", "methyltransferase",
    "glycosyl", "nuclease", "polymerase", "helicase", "ATPase", "GTPase",
    "beta-lactamase", "amidase", "deaminase", "mutase", "epimerase", "racemase",
    "catalytic", "active site", "substrate binding",
]

dark_candidates = []
for _, row in rep.iterrows():
    pct = row.cluster_pct_characterized
    pfams = row.cluster_top_pfam_names
    tier = row.naming_tier

    if pct > 10:  # not truly dark
        continue
    if pfams is None:
        continue
    if tier is None or tier < 3:  # too well-characterized
        continue

    # Check if any Pfam name suggests enzyme activity
    pfam_text = " ".join([str(p[1]) for p in pfams if isinstance(p, tuple) and len(p) > 1]).lower()
    is_enzyme_like = any(kw in pfam_text for kw in enzyme_pfam_keywords)

    if is_enzyme_like:
        dark_candidates.append({
            "protein_hash": row.protein_hash,
            "uniref_accession": row.uniref_match_accession,
            "pfams": pfams,
            "top_phyla": row.top_phyla,
            "product_name": row.product_name,
            "pct_characterized": pct,
            "naming_tier": tier,
            "cluster_mean_domain_coverage": row.cluster_mean_domain_coverage,
        })

print(f"  Dark enzyme-like candidates: {len(dark_candidates)}", flush=True)

# Select diverse candidates across phyla
# Prefer those with UniRef match for sequence retrieval
with_uniref = [d for d in dark_candidates if d["uniref_accession"] and "UPI" not in str(d["uniref_accession"])]
print(f"  With UniRef accession: {len(with_uniref)}", flush=True)

# Sample diverse set: ~5 per major microbial phylum
import random
random.seed(42)

major_phyla = ["Pseudomonadota", "Actinomycetota", "Bacillota", "Bacteroidota",
               "Acidobacteriota", "Cyanobacteriota", "Chloroflexota", "Planctomycetota",
               "Verrucomicrobiota", "Myxococcota"]

by_phylum = {p: [] for p in major_phyla}
other = []
for d in with_uniref:
    phyla = [p[0] for p in d["top_phyla"]] if d["top_phyla"] else []
    found = False
    for p in major_phyla:
        if p in phyla:
            by_phylum[p].append(d)
            found = True
            break
    if not found:
        other.append(d)

selected = []
for p in major_phyla:
    n = min(8, len(by_phylum[p]))
    selected.extend(random.sample(by_phylum[p], n) if n > 0 else [])
selected.extend(random.sample(other, min(20, len(other))))
print(f"  Selected for analysis: {len(selected)} dark candidates", flush=True)

# ---- Step 2: Fetch sequences via UniProt ----
print("[2/6] Fetching sequences via UniProt...", flush=True)

def fetch_uniref_sequence(accession):
    """Get sequence from UniProt by UniRef accession."""
    if not accession or accession == "None":
        return None
    # UniRef90_A0A1B2C3D4 -> A0A1B2C3D4
    if "_" in str(accession):
        uniprot_id = str(accession).split("_", 1)[1]
    else:
        uniprot_id = str(accession)

    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    req = urllib.request.Request(url, headers={"User-Agent": "esmc-natmicro/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            fasta = r.read().decode("utf-8")
            lines = fasta.strip().split("\n")
            seq = "".join(lines[1:])
            return seq if len(seq) >= 80 else None
    except Exception as e:
        return None

dark_proteins = []
for i, d in enumerate(selected):
    seq = fetch_uniref_sequence(d["uniref_accession"])
    if seq and 80 <= len(seq) <= 800:
        dark_proteins.append({**d, "sequence": seq})
    if (i+1) % 10 == 0:
        print(f"  {i+1}/{len(selected)} fetched, {len(dark_proteins)} valid", flush=True)
    time.sleep(0.2)

print(f"  Valid dark proteins with sequences: {len(dark_proteins)}", flush=True)

# ---- Step 3: Extract ESMC-SAE features ----
print(f"[3/6] Extracting ESMC-SAE features for {len(dark_proteins)} dark proteins...", flush=True)

tokenizer = AutoTokenizer.from_pretrained(ESMC_PATH)
sae = AutoModel.from_pretrained(SAE_PATH, torch_dtype=torch.bfloat16, device="cpu")
sae.initialize_layers([SAE_LAYER])
sae_layer = sae.layers[str(SAE_LAYER)]
sae_layer.eval()
model = AutoModel.from_pretrained(ESMC_PATH, torch_dtype=torch.bfloat16, device_map="mps").eval()

dark_seqs = [d["sequence"] for d in dark_proteins]
n_dark = len(dark_seqs)
dark_weights = np.zeros((n_dark, FEATURE_DIM), dtype=np.float32)

t0 = time.time()
for i in range(0, n_dark, BATCH_SIZE):
    batch = dark_seqs[i:i+BATCH_SIZE]
    inputs = tokenizer(batch, return_tensors="pt", padding=True)
    inputs = {k: v.to("mps") for k, v in inputs.items()}
    with torch.inference_mode():
        output = model(**inputs, output_hidden_states=True)
        h60 = output.hidden_states[SAE_LAYER].to("cpu").float()
        sae_out = sae_layer(h60)
        fm = sae_out.feature_magnitudes
        for j in range(fm.shape[0]):
            dark_weights[i+j] = fm[j].mean(dim=0).detach().numpy()
    if (i // BATCH_SIZE) % 20 == 0:
        print(f"  {min(i+BATCH_SIZE, n_dark)}/{n_dark} done", flush=True)

# Binarize
dark_binary = np.zeros_like(dark_weights, dtype=np.uint8)
for i in range(n_dark):
    top_idx = np.argpartition(-dark_weights[i], TOP_K)[:TOP_K]
    dark_binary[i, top_idx] = 1

print(f"  Extraction done in {time.time()-t0:.1f}s", flush=True)

# ---- Step 4: Load trained EC classifier & predict ----
print("[4/6] Loading EC classifier and predicting...", flush=True)

# Load training data
train_weights = np.load(f"{OUT_DIR}/microbial_esmc_sae_weights.npy")
train_binary = np.load(f"{OUT_DIR}/microbial_esmc_sae_binary.npy")
with open(f"{OUT_DIR}/microbial_esmc_sae_meta.pkl", "rb") as f:
    train_meta = pickle.load(f)

le = LabelEncoder()
y_train = le.fit_transform(train_meta["ecs"])
n_classes = len(le.classes_)

# Train classifier on full training set (use binary only for speed)
clf = LogisticRegression(max_iter=300, C=1.0)
clf.fit(train_binary.astype(np.float32), y_train)
print(f"  Trained on {len(y_train)} enzymes, {n_classes} EC4 classes", flush=True)

# Predict on dark proteins
proba = clf.predict_proba(dark_binary.astype(np.float32))
top5_idx = np.argsort(-proba, axis=1)[:, :5]

# Get top-5 EC predictions per dark protein
predictions = []
for i in range(n_dark):
    preds = []
    for j in range(5):
        ec_idx = top5_idx[i, j]
        ec = le.classes_[ec_idx]
        score = proba[i, ec_idx]
        ec1 = ec.split(".")[0]
        preds.append({"ec": ec, "score": float(score), "ec1": ec1})
    predictions.append(preds)

# ---- Step 5: Cross-validate with Pfam ----
print("[5/6] Cross-validating predictions with Pfam annotations...", flush=True)

# Map EC1 to expected Pfam keywords (simplified)
ec1_pfam_map = {
    "1": ["oxidoreductase", "dehydrogenase", "oxidase", "reductase", "monooxygenase", "dioxygenase", "peroxidase", "catalase"],
    "2": ["transferase", "kinase", "methyltransferase", "acetyltransferase", "glycosyltransferase", "aminotransferase", "phosphotransferase"],
    "3": ["hydrolase", "protease", "peptidase", "esterase", "lipase", "phosphatase", "nuclease", "glycosidase", "amidase", "deaminase", "lactamase"],
    "4": ["lyase", "decarboxylase", "dehydratase", "aldolase", "synthase", "cyclase", "deaminase"],
    "5": ["isomerase", "mutase", "epimerase", "racemase", "isomerase", "topoisomerase"],
    "6": ["ligase", "synthetase", "carboxylase"],
    "7": ["translocase", "transporter", "ATPase", "permease"],
}

results = []
for i, d in enumerate(dark_proteins):
    pfam_text = " ".join([str(p[1]) for p in d["pfams"]]).lower() if d["pfams"] else ""
    top_ec1 = predictions[i][0]["ec1"]
    expected_keywords = ec1_pfam_map.get(top_ec1, [])

    pfam_support = any(kw in pfam_text for kw in expected_keywords)
    # Also check if ANY top-5 prediction matches
    any_match = False
    for pred in predictions[i]:
        ek = ec1_pfam_map.get(pred["ec1"], [])
        if any(kw in pfam_text for kw in ek):
            any_match = True
            break

    results.append({
        **d,
        "top_prediction": predictions[i][0],
        "top5_predictions": predictions[i],
        "pfam_supports_top1": pfam_support,
        "pfam_supports_any": any_match,
    })

pfam_support_count = sum(1 for r in results if r["pfam_supports_top1"])
pfam_support_any = sum(1 for r in results if r["pfam_supports_any"])
print(f"  Pfam supports top-1 prediction: {pfam_support_count}/{len(results)} "
      f"({pfam_support_count/len(results)*100:.1f}%)", flush=True)
print(f"  Pfam supports any top-5: {pfam_support_any}/{len(results)} "
      f"({pfam_support_any/len(results)*100:.1f}%)", flush=True)

# ---- Step 6: Generate case study report ----
print("[6/6] Generating report...", flush=True)

# EC1 prediction distribution
ec1_pred = Counter(r["top_prediction"]["ec1"] for r in results)
print(f"\n  EC1 prediction distribution: {dict(sorted(ec1_pred.items()))}")

# Top confidence predictions (score > 0.5)
high_conf = [r for r in results if r["top_prediction"]["score"] > 0.5]
print(f"  High-confidence predictions (score>0.5): {len(high_conf)}")

# Save full results
output = {
    "n_dark_proteins": n_dark,
    "pfam_support_top1_rate": pfam_support_count / len(results) if results else 0,
    "pfam_support_any_rate": pfam_support_any / len(results) if results else 0,
    "ec1_distribution": dict(ec1_pred),
    "high_confidence_count": len(high_conf),
    "results": [{
        "protein_hash": r["protein_hash"],
        "uniref_accession": r["uniref_accession"],
        "product_name": r["product_name"],
        "pfams": [(p[0], p[1]) for p in r["pfams"]] if r["pfams"] else [],
        "top_phyla": r["top_phyla"],
        "pct_characterized": r["pct_characterized"],
        "top_prediction": r["top_prediction"],
        "top5_predictions": r["top5_predictions"],
        "pfam_supports_top1": r["pfam_supports_top1"],
        "pfam_supports_any": r["pfam_supports_any"],
    } for r in sorted(results, key=lambda x: x["top_prediction"]["score"], reverse=True)]
}

with open(f"{OUT_DIR}/dark_matter_results.json", "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

# Print top-10 discoveries
print(f"\n{'='*80}")
print("TOP DARK MATTER ENZYME PREDICTIONS")
print(f"{'='*80}")
for i, r in enumerate(sorted(results, key=lambda x: x["top_prediction"]["score"], reverse=True)[:10]):
    print(f"\n  #{i+1} | Score: {r['top_prediction']['score']:.3f} | "
          f"Predicted EC: {r['top_prediction']['ec']} (EC{r['top_prediction']['ec1']})")
    print(f"  Product: {r['product_name']}")
    print(f"  Pfams: {[p[1] for p in r['pfams'][:5]] if r['pfams'] else 'N/A'}")
    print(f"  Pfam supports: {'YES' if r['pfam_supports_top1'] else 'NO'}")
    print(f"  Phyla: {[p[0] for p in r['top_phyla'][:3]] if r['top_phyla'] else 'N/A'}")

print(f"\nSaved to {OUT_DIR}/dark_matter_results.json")
print("Done!")
