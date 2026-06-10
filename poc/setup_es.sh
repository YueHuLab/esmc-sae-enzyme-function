#!/bin/bash
# Setup Elasticsearch for SAE protein search
# Usage: bash setup_es.sh

set -e

echo "=== Starting Elasticsearch ==="
docker rm -f protein-search 2>/dev/null || true
docker run -d --name protein-search \
  -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  elasticsearch:8.15.0

echo "Waiting for ES to be ready..."
until curl -s http://localhost:9200/_cluster/health | grep -q '"status":"green"\|"status":"yellow"'; do
  sleep 2
  echo "  waiting..."
done
echo "ES is ready!"

echo ""
echo "=== Creating index with mapping ==="
curl -s -X PUT "http://localhost:9200/sae_proteins" -H 'Content-Type: application/json' -d '{
  "mappings": {
    "properties": {
      "protein_id":  { "type": "keyword" },
      "ec_number":   { "type": "keyword" },
      "ec1":         { "type": "keyword" },
      "features":    { "type": "keyword" }
    }
  }
}' | python3 -m json.tool

echo ""
echo "=== Bulk indexing ==="
cd /Users/huyue/esmc_search/poc
python3 -c "
import requests, json, time
t0 = time.time()
with open('data/es_bulk_1210.jsonl', 'rb') as f:
    r = requests.post('http://localhost:9200/_bulk',
                      headers={'Content-Type': 'application/x-ndjson'},
                      data=f.read())
result = json.loads(r.text)
if result.get('errors'):
    print(f'Errors during bulk indexing:')
    for item in result['items']:
        if 'error' in item.get('index', {}):
            print(f'  {item[\"index\"][\"_id\"]}: {item[\"index\"][\"error\"][\"reason\"]}')
else:
    print(f'Indexed {len(result[\"items\"])} documents in {time.time()-t0:.1f}s')

# Refresh
requests.post('http://localhost:9200/sae_proteins/_refresh')
count = json.loads(requests.get('http://localhost:9200/sae_proteins/_count').text)
print(f'Total documents in index: {count[\"count\"]}')
"

echo ""
echo "=== Test query ==="
curl -s -X POST "http://localhost:9200/sae_proteins/_search" -H 'Content-Type: application/json' -d '{
  "query": {
    "bool": {
      "must": [
        {"term": {"features": 500}},
        {"term": {"features": 1200}}
      ]
    }
  },
  "size": 5
}' | python3 -c "
import json, sys
d = json.load(sys.stdin)
print(f'Hits: {d[\"hits\"][\"total\"][\"value\"]}')
for hit in d['hits']['hits']:
    s = hit['_source']
    print(f'  {s[\"protein_id\"]:10s}  EC={s[\"ec_number\"]:12s}  EC1={s[\"ec1\"]}')
print(f'Took: {d[\"took\"]}ms')
"

echo ""
echo "=== Setup complete ==="
