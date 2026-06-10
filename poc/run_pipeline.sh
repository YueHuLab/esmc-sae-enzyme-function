#!/bin/bash
# Full pipeline for Nature Microbiology EC prediction paper
# Run after extract_microbial_sae.py completes

set -e
cd /Users/huyue/esmc_search/poc

echo "============================================"
echo "Nature Microbiology EC Prediction Pipeline"
echo "============================================"
echo "Started at: $(date)"
echo ""

# Step 1: Benchmark
echo "=== Step 1/3: EC Prediction Benchmark ==="
python3 -u benchmark_microbial.py
echo "Benchmark done at: $(date)"
echo ""

# Step 2: Dark Matter Analysis
echo "=== Step 2/3: Dark Matter Enzyme Discovery ==="
python3 -u dark_matter_analysis.py
echo "Dark matter done at: $(date)"
echo ""

# Step 3: Interpretability
echo "=== Step 3/3: Feature Interpretability ==="
python3 -u interpretability.py
echo "Interpretability done at: $(date)"
echo ""

echo "============================================"
echo "Pipeline complete at: $(date)"
echo ""
echo "Output files:"
echo "  results/benchmark_microbial.json"
echo "  results/dark_matter_results.json"
echo "  results/interpretability_results.json"
echo "  plots/benchmark_microbial.png"
echo "  plots/interpretability_ec1.png"
echo "============================================"
