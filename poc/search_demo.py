"""
Full SAE search engine demo: inverted index + feature descriptions + ES bulk export.

Usage:
  python3 search_demo.py                  # interactive search
  python3 search_demo.py --export-es       # export ES bulk JSON
  python3 search_demo.py --eval            # evaluate precision/recall on EC classes
"""
import numpy as np
import pandas as pd
import pickle
import json
import time
import sys

DATA_DIR = "data"
OUT_DIR = "results"
FEATURE_TABLE = "/Users/huyue/esm-c-fold2/ESMC-SAE-Features/uniref90_feature_table.parquet"

class SAESearchEngine:
    def __init__(self):
        print("Loading feature descriptions...", flush=True)
        self.features_df = pd.read_parquet(FEATURE_TABLE)
        self.features_df = self.features_df.set_index("feature_id")
        print(f"  {len(self.features_df)} features, {len(self.features_df['category'].unique())} categories")

        print("Loading protein data...", flush=True)
        self.binary = np.load(f"{OUT_DIR}/esmc_sae_binary.npy")
        with open(f"{OUT_DIR}/meta.pkl", "rb") as f:
            self.meta = pickle.load(f)
        self.protein_ids = self.meta["accs"]
        self.ec_numbers = self.meta["ecs"]
        self.ec1_list = self.meta["ec1"]
        self.n_proteins, self.n_features = self.binary.shape

        # IDF from feature table (higher quality than Atlas normalization)
        self.idf = self.features_df["uniref90_idf"].values.astype(np.float32)

        print("Building inverted index...", flush=True)
        t0 = time.time()
        self.posting = {}
        for f in range(self.n_features):
            idx = np.where(self.binary[:, f])[0]
            if len(idx) > 0:
                self.posting[f] = idx.tolist()
        print(f"  {len(self.posting)}/{self.n_features} non-empty, {time.time()-t0:.2f}s")

    def search_features(self, keywords, category=None, top_k=20):
        """Search for feature IDs by keyword in description/summary, optionally filtered by category."""
        results = []
        kw_lower = keywords.lower()
        for fid in range(self.n_features):
            if fid not in self.features_df.index:
                continue
            row = self.features_df.loc[fid]
            if category and row["category"] != category:
                continue
            summary = str(row["summary"]).lower()
            if kw_lower in summary:
                results.append({
                    "feature_id": fid,
                    "summary": row["summary"],
                    "category": row["category"],
                    "frequency": int(row["uniref90_frequency"]),
                    "idf": float(row["uniref90_idf"]),
                })
        results.sort(key=lambda x: x["idf"], reverse=True)
        return results[:top_k]

    def query(self, must=None, should=None, must_not=None, top_k=20, verbose=True):
        """Boolean search over features. Returns scored protein matches."""
        must = must or []; should = should or []; must_not = must_not or []
        t0 = time.time()

        # Resolve feature IDs (if they're summaries, find matching IDs)
        result_set = set(range(self.n_proteins))

        for fid in must:
            if fid in self.posting:
                result_set &= set(self.posting[fid])
            else:
                result_set = set(); break

        for fid in must_not:
            if fid in self.posting:
                result_set -= set(self.posting[fid])

        if should:
            should_set = set()
            for fid in should:
                if fid in self.posting:
                    should_set |= set(self.posting[fid])
            result_set &= should_set

        # Score by IDF-weighted feature matches
        scored = []
        for idx in result_set:
            vec = self.binary[idx]
            score = sum(self.idf[fid] for fid in must + should if vec[fid])
            scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        elapsed_ms = (time.time() - t0) * 1000

        if verbose:
            print(f"\nQuery: must={must}, should={should}, must_not={must_not}")
            for fid in must + should + must_not:
                if fid < len(self.features_df):
                    print(f"  [{fid}] {self.features_df.loc[fid, 'summary'][:100]}")
            print(f"  → {len(scored)} matches in {elapsed_ms:.2f}ms")
            for rank, (idx, score) in enumerate(scored[:min(top_k, len(scored))]):
                print(f"  #{rank+1}: {self.protein_ids[idx]:10s}  EC={self.ec_numbers[idx]:12s}  "
                      f"EC1={self.ec1_list[idx]}  score={score:.1f}")

        return scored[:top_k]

    def evaluate_ec_retrieval(self, ec_class, n_query_features=10, top_k=100):
        """Given an EC class, find its most discriminative features, query, measure recall."""
        # Find all proteins with this EC class
        target_idx = [i for i, ec in enumerate(self.ec_numbers) if ec == ec_class]
        if len(target_idx) < 5:
            return None

        # Most discriminative features: high precision for this EC class
        target_vecs = self.binary[target_idx]
        non_target_vecs = np.delete(self.binary, target_idx, axis=0)

        # Feature enrichment in target vs background
        target_rate = target_vecs.mean(axis=0)
        bg_rate = non_target_vecs.mean(axis=0)
        # Simple lift: target_rate / (bg_rate + eps), weighted by target_rate
        eps = 1e-5
        lift = target_rate / (bg_rate + eps)
        # Score: lift * target_rate (prefer features common in target but rare overall)
        score = lift * target_rate
        top_features = np.argsort(-score)[:n_query_features].tolist()

        # Query with these features (OR logic)
        t0 = time.time()
        result_set = set()
        for fid in top_features:
            if fid in self.posting:
                result_set |= set(self.posting[fid])

        found = len(result_set & set(target_idx))
        recall = found / len(target_idx)
        precision = found / len(result_set) if result_set else 0
        elapsed_ms = (time.time() - t0) * 1000

        return {
            "ec_class": ec_class,
            "n_target": len(target_idx),
            "n_retrieved": len(result_set),
            "found": found,
            "recall": recall,
            "precision": precision,
            "time_ms": elapsed_ms,
            "features": top_features,
        }

    def evaluate_all_ec(self, n_query_features=10, min_target=10):
        """Evaluate precision/recall for all EC classes with enough samples."""
        from collections import Counter
        ec_counts = Counter(self.ec_numbers)
        results = []
        for ec, count in ec_counts.most_common():
            if count < min_target:
                continue
            r = self.evaluate_ec_retrieval(ec, n_query_features=n_query_features, top_k=200)
            if r:
                results.append(r)
        return results

    def export_es_bulk(self, out_path="data/es_bulk_1210.jsonl"):
        """Export proteins with features to Elasticsearch bulk JSONL format."""
        print(f"Exporting ES bulk to {out_path}...", flush=True)
        with open(out_path, "w") as f:
            for i in range(self.n_proteins):
                # Index action
                action = {"index": {"_index": "sae_proteins", "_id": self.protein_ids[i]}}
                f.write(json.dumps(action) + "\n")
                # Document
                active_features = np.where(self.binary[i])[0].tolist()
                doc = {
                    "protein_id": self.protein_ids[i],
                    "ec_number": self.ec_numbers[i],
                    "ec1": self.ec1_list[i],
                    "features": active_features,  # keyword array for ES inverted index
                }
                f.write(json.dumps(doc) + "\n")
        size_mb = len(open(out_path).read()) / 1e6
        print(f"  Exported {self.n_proteins} proteins, {size_mb:.1f} MB")

    def es_query_template(self, must=None, should=None, must_not=None, size=20):
        """Generate an ES query JSON for the given feature selection."""
        query = {"query": {"bool": {}}, "size": size}
        if must:
            query["query"]["bool"]["must"] = [
                {"term": {"features": fid}} for fid in must
            ]
        if should:
            query["query"]["bool"]["should"] = [
                {"term": {"features": fid}} for fid in should
            ]
        if must_not:
            query["query"]["bool"]["must_not"] = [
                {"term": {"features": fid}} for fid in must_not
            ]
        return query


if __name__ == "__main__":
    engine = SAESearchEngine()

    if "--export-es" in sys.argv:
        engine.export_es_bulk()
        # Also print example ES query
        print("\nExample ES query:")
        print(json.dumps(engine.es_query_template(
            must=[500, 1200], should=[3000], must_not=[10000]
        ), indent=2))

    elif "--eval" in sys.argv:
        print("\nEvaluating EC retrieval...")
        results = engine.evaluate_all_ec(n_query_features=10, min_target=10)
        recalls = [r["recall"] for r in results]
        precisions = [r["precision"] for r in results]
        times = [r["time_ms"] for r in results]
        print(f"\n{len(results)} EC classes evaluated")
        print(f"  Mean recall:    {np.mean(recalls):.3f}")
        print(f"  Mean precision: {np.mean(precisions):.3f}")
        print(f"  Mean time:      {np.mean(times):.1f}ms")
        # Show best and worst
        results.sort(key=lambda x: x["recall"], reverse=True)
        print("\nTop-5 by recall:")
        for r in results[:5]:
            print(f"  {r['ec_class']:12s}  recall={r['recall']:.3f}  "
                  f"precision={r['precision']:.3f}  found={r['found']}/{r['n_target']}  "
                  f"{r['time_ms']:.1f}ms")

    else:
        # Interactive demo
        print("\n" + "=" * 60)
        print("FEATURE SEARCH EXAMPLES")
        print("=" * 60)

        # Show feature categories
        cats = engine.features_df["category"].value_counts()
        print("\nAvailable categories:")
        for cat, count in cats.items():
            print(f"  {cat:35s} {count:5d} features")

        # Search for interesting features
        for kw in ["catalytic triad", "alpha/beta hydrolase", "zinc binding", "signal peptide",
                   "transmembrane helix", "disordered", "kinase", "TIM barrel"]:
            results = engine.search_features(kw, top_k=3)
            print(f"\nSearch '{kw}': found {len(results)} features")
            for r in results[:3]:
                print(f"  [{r['feature_id']:5d}] ({r['category']:25s}) {r['summary'][:120]}")

        # Boolean query demo
        print("\n" + "=" * 60)
        print("BOOLEAN QUERY DEMO")
        print("=" * 60)

        # Find hydrolase-related features
        hydrolase_features = engine.search_features("hydrolase", top_k=5)
        if len(hydrolase_features) >= 3:
            engine.query(
                must=[hydrolase_features[0]["feature_id"]],
                should=[f["feature_id"] for f in hydrolase_features[1:3]],
                top_k=10,
            )

        # Evaluate a few EC classes
        print("\n" + "=" * 60)
        print("EC EVALUATION SAMPLE")
        print("=" * 60)
        sample_ecs = ["2.7.11.1", "3.1.3.48", "1.1.1.1", "6.1.1.1"]  # common ECs
        for ec in sample_ecs:
            r = engine.evaluate_ec_retrieval(ec, n_query_features=10)
            if r:
                print(f"  {ec:12s}  recall={r['recall']:.3f}  precision={r['precision']:.3f}  "
                      f"found={r['found']}/{r['n_target']}  retrieved={r['n_retrieved']}  "
                      f"{r['time_ms']:.1f}ms")
