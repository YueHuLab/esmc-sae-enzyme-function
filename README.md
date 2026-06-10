# ESMC-SAE Enzyme Function Prediction

[![bioRxiv](https://img.shields.io/badge/bioRxiv-preprint-b8453e)](https://biorxiv.org)
[![HuggingFace](https://img.shields.io/badge/🤗-YueHuLab%2Fesmc--sae--enzyme--function-blue)](https://huggingface.co/YueHuLab/esmc-sae-enzyme-function)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Interpretable enzyme function prediction via sparse autoencoder features of ESMC across the microbial protein universe**

Yue Hu, Junqing Wang, Yingchao Liu

---

## Overview

This repository contains the code and data for predicting enzyme commission (EC) numbers using sparse autoencoder (SAE) features extracted from ESMC-6B, a 6-billion-parameter protein language model. The ESMC SAE decomposes protein representations into 16,384 independently interpretable biological concepts (catalytic sites, binding pockets, structural motifs, etc.), each annotated by GPT-5.

**Key findings:**
- **78.9% top-1 EC3 accuracy** (161 classes, 4,868 microbial enzymes), +37.6% over sequence baselines
- **47.7% leave-one-class-out EC1 recovery** (3.3× random baseline), demonstrating generalization to novel enzyme classes
- **169,859 dark enzyme-like candidates** identified in the ESM Atlas for experimental follow-up
- Features driving predictions are **mechanistically interpretable** (e.g., catalytic triad geometry → hydrolases)

## Installation

```bash
# Clone
git clone https://github.com/YueHuLab/esmc-sae-enzyme-function.git
cd esmc-sae-enzyme-function

# Install dependencies
pip install torch transformers scikit-learn pandas matplotlib

# Download ESMC-6B model (25 GB) and SAE from HuggingFace
# Models available at: https://huggingface.co/biohub/ESMC-6B
# SAE: https://huggingface.co/biohub/ESMC-6B-sae-layer60-k64-codebook16384
```

## Quick Start

```bash
cd poc

# 1. Fetch microbial enzyme data from UniProt
python fetch_microbial_enzymes.py

# 2. Extract ESMC-SAE features
python extract_microbial_sae.py

# 3. Run benchmark (80/20 + leave-one-class-out)
python benchmark_microbial.py

# 4. Interpretability analysis
python interpretability.py

# 5. Dark matter survey
python dark_matter_analysis.py
```

## Repository Structure

```
├── paper/
│   ├── manuscript.tex          # LaTeX manuscript
│   ├── manuscript.pdf          # Compiled PDF
│   ├── references.bib          # Bibliography
│   └── figures/                # Publication figures
├── poc/
│   ├── fetch_microbial_enzymes.py   # UniProt data collection
│   ├── extract_microbial_sae.py     # ESMC-SAE feature extraction
│   ├── benchmark_microbial.py       # EC prediction benchmark
│   ├── loco_quick.py               # Leave-one-class-out evaluation
│   ├── interpretability.py          # Feature-EC interpretability
│   ├── dark_matter_analysis.py      # Dark enzyme survey
│   ├── generate_figures.py          # Paper figure generation
│   ├── search_demo.py               # Feature search demo
│   └── inverted_index.py            # Inverted index for Boolean queries
└── PROPOSAL.md                 # Original project proposal
```

## Data

- **Training**: 4,868 microbial SwissProt enzymes (Bacteria + Archaea) with complete EC annotations
- **Features**: 16,384-dimensional SAE activations (binary: top-64 per protein; weights: mean-pooled)
- **Dark matter**: 367,956 dark clusters from ESM Atlas (7.7M representatives)
- **Source**: UniProt REST API + ESM Atlas AWS Open Data (`s3://esm-protein-atlas/v1/`)

Pre-computed features and benchmarks available on [HuggingFace](https://huggingface.co/YueHuLab/esmc-sae-enzyme-function).

## Citation

```bibtex
@article{hu2026interpretable,
  title={Interpretable enzyme function prediction via sparse autoencoder features of ESMC across the microbial protein universe},
  author={Hu, Yue and Wang, Junqing and Liu, Yingchao},
  journal={bioRxiv},
  year={2026}
}
```

## License

MIT License.
