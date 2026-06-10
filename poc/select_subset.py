"""
Select a well-stratified subset of microbial enzymes for the Nature Microbiology paper.
Target: ~5,000 enzymes balanced across EC1 classes, with good EC4 diversity.
Also filters to moderate lengths for extraction speed.
"""
import random
from collections import Counter

random.seed(42)

DATA = "data/microbial_enzymes.tsv"
OUT = "data/microbial_enzymes_5k.tsv"

print("Loading full dataset...")
with open(DATA) as f:
    header = f.readline().strip()
    lines = [line.strip() for line in f if line.strip()]

print(f"Total: {len(lines)} enzymes")

# Parse
acc_idx = header.split("\t").index("Entry")
ec_idx = header.split("\t").index("EC number")
seq_idx = header.split("\t").index("Sequence")

# Filter by length (100-600aa for extraction speed)
filtered = []
for line in lines:
    fields = line.split("\t")
    seq = fields[seq_idx]
    if 80 <= len(seq) <= 700:
        filtered.append(line)

print(f"After length filter (80-700aa): {len(filtered)}")

# Group by EC3 (first 3 levels) for stratification
ec3_groups = {}
for line in filtered:
    fields = line.split("\t")
    ec = fields[ec_idx]
    ec3 = ".".join(ec.split(".")[:3])
    if ec3 not in ec3_groups:
        ec3_groups[ec3] = []
    ec3_groups[ec3].append(line)

print(f"EC3 classes: {len(ec3_groups)}")

# Sample per EC3: at most 30 per EC3 (ensures diversity)
# EC1 totals: ~700-800 each for balanced classes
balanced = []
ec1_counts = Counter()
for ec3, items in sorted(ec3_groups.items()):
    n_sample = min(55, len(items))
    sampled = random.sample(items, n_sample)
    balanced.extend(sampled)
    ec1 = ec3.split(".")[0]
    ec1_counts[ec1] += n_sample

print(f"\nBalanced subset: {len(balanced)} enzymes")
print(f"EC1 distribution: {dict(sorted(ec1_counts.items()))}")

# Count EC3 classes
ec3_final = Counter()
for line in balanced:
    ec = line.split("\t")[ec_idx]
    ec3 = ".".join(ec.split(".")[:3])
    ec3_final[ec3] += 1
print(f"EC3 classes: {len(ec3_final)}")

# Save
with open(OUT, "w") as f:
    f.write(header + "\n")
    for line in balanced:
        f.write(line + "\n")

print(f"\nSaved {len(balanced)} enzymes to {OUT}")
