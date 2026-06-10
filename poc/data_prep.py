"""
Download SwissProt enzymes with EC numbers via UniProt REST API.
Filters to high-quality (annotation_score=5), experimentally curated.
Saves to data/swissprot_enzymes.tsv
"""
import urllib.request
import urllib.parse
import time
import csv
import sys
import os
import re

BASE = "https://rest.uniprot.org/uniprotkb/search"
FIELDS = "accession,id,protein_name,ec,sequence,annotation_score"

QUERY = (
    "(reviewed:true) AND (ec:*) AND (annotation_score:5) AND "
    "(length:[100 TO 800])"
)


def fetch_page(cursor_url=None, size=500):
    if cursor_url is None:
        params = {
            "query": QUERY,
            "format": "tsv",
            "fields": FIELDS,
            "size": str(size),
        }
        url = BASE + "?" + urllib.parse.urlencode(params)
    else:
        url = cursor_url
    req = urllib.request.Request(url, headers={"User-Agent": "esmc-poc/1.0"})
    with urllib.request.urlopen(req, timeout=60) as r:
        body = r.read().decode("utf-8")
        link = r.headers.get("Link", "")
    next_url = None
    if link:
        m = re.search(r'<([^>]+)>\s*;\s*rel="next"', link)
        if m:
            next_url = m.group(1)
    return body, next_url


def main(target=5000, out_path="data/swissprot_enzymes.tsv"):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    cursor = None
    while len(rows) < target:
        body, cursor = fetch_page(cursor_url=cursor, size=500)
        lines = body.strip().split("\n")
        if len(lines) <= 1 and not cursor:
            break
        # skip header on first page only
        if not rows:
            rows.append(lines[0])
        for line in lines[1:]:
            if not line.strip():
                continue
            rows.append(line)
        print(f"  fetched {len(rows)-1} / {target}  (next={'yes' if cursor else 'no'})", flush=True)
        if not cursor:
            break
        time.sleep(0.4)
    with open(out_path, "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"saved {len(rows)-1} entries to {out_path}")


if __name__ == "__main__":
    main(target=int(sys.argv[1]) if len(sys.argv) > 1 else 5000)
