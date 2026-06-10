"""
Fetch microbial (Bacteria + Archaea) SwissProt enzymes with complete EC numbers.
Target: ~15,000 enzymes with diverse EC classes for Nature Microbiology paper.

Strategy: fetch by EC1 class separately to ensure balanced coverage.
"""
import urllib.request, urllib.parse, time, os, re, json
from collections import Counter

OUT_DIR = "data"
FIELDS = "accession,id,protein_name,ec,sequence,organism_name,lineage,annotation_score,ft_domain"
BASE = "https://rest.uniprot.org/uniprotkb/search"

os.makedirs(OUT_DIR, exist_ok=True)


def fetch_count(query):
    """Get total count for a query."""
    params = {"query": query, "format": "json", "size": "1"}
    url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "esmc-natmicro/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers.get("x-total-results", 0))


def fetch_page(query, cursor_url=None, fields=FIELDS, size=500):
    if cursor_url:
        url = cursor_url
    else:
        params = {"query": query, "format": "tsv", "fields": fields, "size": str(size)}
        url = BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "esmc-natmicro/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8")
        link = r.headers.get("Link", "")
    next_url = None
    if link:
        m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        if m:
            next_url = m.group(1)
    return body, next_url


def fetch_all(query, max_count=None):
    """Fetch all results for a query, with optional cap."""
    rows = []
    cursor = None
    while True:
        body, cursor = fetch_page(query, cursor_url=cursor)
        lines = body.strip().split("\n")
        if not rows:
            rows.append(lines[0])  # header
        for line in lines[1:]:
            if line.strip():
                rows.append(line)
        print(f"  {len(rows)-1} rows fetched", flush=True)
        if not cursor:
            break
        if max_count and len(rows) - 1 >= max_count:
            break
        time.sleep(0.3)
    return rows


# Base query: microbial (Bacteria + Archaea), reviewed, with EC, reasonable length
BASE_Q = "(reviewed:true) AND (ec:*) AND (taxonomy_name:Bacteria OR taxonomy_name:Archaea) AND (length:[80 TO 1000])"

# EC1 classes to cover (use wildcard patterns for UniProt query)
EC1_CLASSES = [
    ("EC1", "Oxidoreductases", "1.*"),
    ("EC2", "Transferases", "2.*"),
    ("EC3", "Hydrolases", "3.*"),
    ("EC4", "Lyases", "4.*"),
    ("EC5", "Isomerases", "5.*"),
    ("EC6", "Ligases", "6.*"),
    ("EC7", "Translocases", "7.*"),
]

print("=" * 60)
print("Fetching microbial enzyme data for Nature Microbiology paper")
print("=" * 60)

# Phase 1: Count per EC1 class
print("\n[1] Counting enzymes per EC1 class...")
ec1_counts = {}
for ec1, name, pattern in EC1_CLASSES:
    q = f"{BASE_Q} AND (ec:{pattern})"
    n = fetch_count(q)
    ec1_counts[ec1] = n
    print(f"  {ec1} {name}: {n:,}")

total = sum(ec1_counts.values())
print(f"  TOTAL: {total:,}")

# Phase 2: Fetch high-quality subset, stratified by EC1
# Target ~2,000 per EC1 class (EC4/EC6 are rarer, take what we can)
TARGETS = {"EC1": 1500, "EC2": 2500, "EC3": 2500, "EC4": 1500,
           "EC5": 1500, "EC6": 1500, "EC7": 1500}

print("\n[2] Fetching stratified by EC1 class (high quality)...")
all_rows = []
header_written = False

for ec1, name, pattern in EC1_CLASSES:
    target = TARGETS.get(ec1, 1500)
    available = ec1_counts.get(ec1, 0)
    q = f"{BASE_Q} AND (ec:{pattern})"
    print(f"\n  --- {ec1} {name} (target: {target}, available: {available}) ---")
    rows = fetch_all(q, max_count=target)
    if not header_written:
        all_rows.append(rows[0])
        header_written = True
    all_rows.extend(rows[1:])
    actual = len(rows) - 1
    print(f"  -> got {actual} for {ec1}")

print(f"\n  TOTAL fetched: {len(all_rows)-1}")

# Phase 3: Clean and deduplicate
print("\n[3] Cleaning...")

header = all_rows[0].split("\t")
acc_idx = header.index("Entry")
ec_idx = header.index("EC number")
seq_idx = header.index("Sequence")

seen = set()
clean = [all_rows[0]]
stats = {"total": 0, "dup": 0, "short_ec": 0, "has_x": 0, "kept": 0}

for line in all_rows[1:]:
    stats["total"] += 1
    fields = line.split("\t")
    if len(fields) < len(header):
        continue
    acc = fields[acc_idx]
    ec = fields[ec_idx]
    seq = fields[seq_idx]

    if acc in seen:
        stats["dup"] += 1
        continue

    # Check EC has at least 3 levels (e.g., 1.2.3.-)
    ec_parts = ec.split(".")
    if len(ec_parts) < 4 or ec_parts[2] == "-":
        stats["short_ec"] += 1
        continue

    # Remove sequences with ambiguous residues
    if "X" in seq or "U" in seq:
        stats["has_x"] += 1
        continue

    seen.add(acc)
    clean.append(line)
    stats["kept"] += 1

print(f"  Total: {stats['total']}, Duplicates: {stats['dup']}, "
      f"Short EC: {stats['short_ec']}, Has X/U: {stats['has_x']}, "
      f"KEPT: {stats['kept']}")

# Save
out_path = f"{OUT_DIR}/microbial_enzymes.tsv"
with open(out_path, "w") as f:
    f.write("\n".join(clean) + "\n")
print(f"\n  Saved {stats['kept']} enzymes to {out_path}")

# Print EC distribution
print("\n[4] Final EC distribution...")
ec1_dist = Counter()
ec4_counts = Counter()
for line in clean[1:]:
    fields = line.split("\t")
    ec = fields[ec_idx]
    ec1 = ec.split(".")[0]
    ec1_dist[ec1] += 1
    ec4 = ".".join(ec.split(".")[:3])
    ec4_counts[ec4] += 1

for ec1, name, pattern in EC1_CLASSES:
    n = ec1_dist.get(ec1[2], 0)
    print(f"  {ec1} {name}: {n}")

print(f"  Unique EC4 classes: {len(ec4_counts)}")
print(f"  Top-10 EC4 classes: {ec4_counts.most_common(10)}")

print("\nDone!")
