"""
In-memory inverted index for ESMC-SAE features.
feature_id → sorted list of protein indices (posting list).
Supports boolean queries: must (AND), should (OR), must_not (NOT).
"""
import numpy as np
import pickle
import time

class SAEInvertedIndex:
    def __init__(self, binary_features, protein_ids, feature_weights=None):
        """
        Args:
            binary_features: (N_proteins, 16384) uint8 binary matrix
            protein_ids: list of protein accession IDs
            feature_weights: optional (16384,) float array of IDF-like weights
        """
        self.binary = binary_features
        self.protein_ids = list(protein_ids)
        self.n_proteins, self.n_features = binary_features.shape
        self.feature_weights = feature_weights
        self._build_index()

    def _build_index(self):
        """Build inverted index: feature → list of protein indices."""
        t0 = time.time()
        self.posting_lists = {}
        # For each feature, find which proteins have it active
        for f in range(self.n_features):
            proteins_with_f = np.where(self.binary[:, f])[0]
            if len(proteins_with_f) > 0:
                self.posting_lists[f] = proteins_with_f.tolist()

        n_nonempty = len(self.posting_lists)
        avg_len = sum(len(v) for v in self.posting_lists.values()) / max(n_nonempty, 1)
        print(f"Index built: {n_nonempty}/{self.n_features} non-empty features, "
              f"avg posting list length={avg_len:.1f}, "
              f"{time.time()-t0:.2f}s", flush=True)

    def _feature_stats(self, feature_ids):
        """Print stats for a list of feature IDs."""
        for fid in feature_ids:
            if fid in self.posting_lists:
                n = len(self.posting_lists[fid])
                idf = self.feature_weights[fid] if self.feature_weights is not None else np.nan
                print(f"  feature {fid:5d}: {n:5d} proteins, idf={idf:.2f}")

    def search(self, must=None, should=None, must_not=None, top_k=50, verbose=True):
        """
        Boolean query over features.

        Args:
            must: list of feature IDs that ALL must be present (AND)
            should: list of feature IDs where AT LEAST ONE must be present (OR)
            must_not: list of feature IDs that must NOT be present (NOT)
            top_k: max results to return
            verbose: print query plan

        Returns:
            list of (protein_index, score) tuples, sorted by score descending
        """
        t0 = time.time()
        must = must or []
        should = should or []
        must_not = must_not or []

        if verbose:
            print(f"\nQuery: must={must}, should={should}, must_not={must_not}")
            print("Feature stats:")
            self._feature_stats(must + should + must_not)

        # Start with all proteins
        result_set = set(range(self.n_proteins))

        # Apply must (AND): intersect result sets
        for fid in must:
            if fid in self.posting_lists:
                result_set &= set(self.posting_lists[fid])
            else:
                result_set = set()  # feature never activates → empty
                break

        # Apply must_not (NOT): remove proteins that have these features
        for fid in must_not:
            if fid in self.posting_lists:
                result_set -= set(self.posting_lists[fid])

        # Apply should (OR): filter to proteins that have at least one should feature
        if should:
            should_set = set()
            for fid in should:
                if fid in self.posting_lists:
                    should_set |= set(self.posting_lists[fid])
            result_set &= should_set

        # Score results: number of (must + should) features matched, weighted by IDF
        results = []
        for idx in result_set:
            protein_vec = self.binary[idx]
            score = 0.0
            for fid in must + should:
                if protein_vec[fid]:
                    w = self.feature_weights[fid] if self.feature_weights is not None else 1.0
                    score += w
            results.append((idx, score))

        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)

        elapsed_ms = (time.time() - t0) * 1000
        if verbose:
            print(f"  Results: {len(results)} proteins matched in {elapsed_ms:.2f}ms")
            if results:
                top_indices = [r[0] for r in results[:min(top_k, len(results))]]
                print(f"  Top {min(top_k, len(results))} protein IDs: {[self.protein_ids[i] for i in top_indices]}")

        return results[:top_k]

    def jaccard_query(self, query_features, top_k=50):
        """
        Jaccard similarity query: given a set of feature IDs,
        find proteins with highest Jaccard overlap.
        """
        t0 = time.time()
        query_set = set(query_features)
        scores = []
        for i in range(self.n_proteins):
            protein_features = set(np.where(self.binary[i])[0])
            intersection = len(query_set & protein_features)
            union = len(query_set | protein_features)
            if union > 0:
                scores.append((i, intersection / union))

        scores.sort(key=lambda x: x[1], reverse=True)
        elapsed_ms = (time.time() - t0) * 1000
        print(f"Jaccard query: {len(query_features)} features → "
              f"{len([s for s in scores if s[1] > 0])} matches in {elapsed_ms:.1f}ms")
        return scores[:top_k]


if __name__ == "__main__":
    # Load data
    print("Loading features...", flush=True)
    binary = np.load("results/esmc_sae_binary.npy")  # (1210, 16384)
    with open("results/meta.pkl", "rb") as f:
        meta = pickle.load(f)
    protein_ids = meta["accs"]
    ec_numbers = meta["ecs"]

    # Load IDF weights
    try:
        with open("data/max_idf_log10.pkl", "rb") as f:
            norm = pickle.load(f)
        idf = norm["idf_per_feature"]
        print(f"Loaded IDF weights: {idf.shape}")
    except FileNotFoundError:
        idf = None
        print("No IDF weights found, using uniform")

    # Build index
    print(f"\nBuilding index for {len(protein_ids)} proteins, {binary.shape[1]} features...")
    idx = SAEInvertedIndex(binary, protein_ids, feature_weights=idf)

    # ---- Demo queries ----
    print("\n" + "=" * 60)
    print("DEMO 1: Query with 3 random features (any protein)")
    # Pick features that appear in at least a few proteins
    active_counts = binary.sum(axis=0)
    # Features present in 5-10% of proteins
    mid_features = np.where((active_counts >= 60) & (active_counts <= 120))[0]
    if len(mid_features) >= 3:
        q = mid_features[:3].tolist()
        results = idx.search(must=q, top_k=10)
        if results:
            best = results[0]
            print(f"  Best match: {protein_ids[best[0]]} (EC={ec_numbers[best[0]]}, score={best[1]:.1f})")

    print("\n" + "=" * 60)
    print("DEMO 2: Transferase-specific query")
    # Find features that are enriched in transferases (EC 2.x.x.x)
    ec1_list = meta["ec1"]
    transferase_idx = [i for i, ec1 in enumerate(ec1_list) if ec1 == "2"]
    hydrolase_idx = [i for i, ec1 in enumerate(ec1_list) if ec1 == "3"]

    if transferase_idx and hydrolase_idx:
        # Features over-represented in transferases vs hydrolases
        trans_mean = binary[transferase_idx].mean(axis=0)
        hydro_mean = binary[hydrolase_idx].mean(axis=0)
        # Simple enrichment: at least 3x more common in transferases
        enrich = np.where((trans_mean > 0.3) & (trans_mean > 3 * hydro_mean))[0]
        print(f"  Found {len(enrich)} transferase-enriched features")
        if len(enrich) >= 3:
            q = enrich[:5].tolist()
            results = idx.search(must=q[:2], should=q[2:5], top_k=10)
            if results:
                best = results[0]
                print(f"  Best match: {protein_ids[best[0]]} (EC={ec_numbers[best[0]]}, "
                      f"EC1={ec1_list[best[0]]})")

    print("\n" + "=" * 60)
    print("DEMO 3: Jaccard query — find proteins similar to a transferase")
    # Use features from the first transferase as query
    if transferase_idx:
        query_features = set(np.where(binary[transferase_idx[0]])[0])
        results = idx.jaccard_query(query_features, top_k=10)
        for i, (pi, score) in enumerate(results[:5]):
            print(f"  #{i+1}: {protein_ids[pi]} EC={ec_numbers[pi]} Jaccard={score:.3f}")

    print("\n" + "=" * 60)
    print("DEMO 4: Boolean with must_not — exclude features")
    if len(mid_features) >= 5:
        q_must = mid_features[:2].tolist()
        q_not = [mid_features[2]]
        results = idx.search(must=q_must, must_not=q_not, top_k=10)

    print("\nIndex ready for interactive queries.")
