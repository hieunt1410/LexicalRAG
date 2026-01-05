import argparse
import json
import os
import pickle
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi
from tqdm import tqdm


class BM25:
    def __init__(
        self,
        queries,
        corpus,
        benchmark="hotpotqa",
        cache_dir="cache",
        n_workers=4,
    ):
        self.queries = queries
        self.corpus = corpus
        self.benchmark = benchmark
        self.cache_dir = cache_dir
        self.n_workers = n_workers
        self.scores = {}

    def build_bm25(self, dataset_name):
        """Build BM25 index for a dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Tuple of (BM25Okapi model, list of doc_ids)
        """
        doc_ids = []
        tokenized_corpus = []
        
        # corpus[dataset_name] is a dict {doc_id: text}
        corpus_data = self.corpus[dataset_name]
        for doc_id, text in tqdm(corpus_data.items(), desc=f"Building BM25 for {dataset_name}"):
            doc_ids.append(doc_id)
            tokenized_corpus.append(text.split())
        return BM25Okapi(tokenized_corpus), doc_ids

    def process_single_query(self, query, query_id, top_k, bm25_model, doc_ids):
        """Process a single query and return top-k results."""
        tokenized_query = query.split()
        scores_array = bm25_model.get_scores(tokenized_query)

        # Use argpartition for O(n) selection instead of O(n log n) sort
        if len(scores_array) > top_k:
            # Get indices of top-k scores efficiently
            top_indices = np.argpartition(scores_array, -top_k)[-top_k:]
            # Sort only the top-k results
            top_indices = top_indices[np.argsort(scores_array[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores_array)[::-1]

        return query_id, [
            (doc_ids[idx], float(scores_array[idx])) for idx in top_indices
        ]

    def get_scores(self, top_k=10, questions=None):
        """Get BM25 scores with parallel processing.
        
        Args:
            top_k: Number of top documents to retrieve
            questions: Optional queries dict (unused, queries are passed in __init__)
            
        Returns:
            Dict mapping dataset_name to {query_id: [(doc_id, score), ...]}
        """
        for dataset_name in self.queries.keys():
            bm25_model, doc_ids = self.build_bm25(dataset_name)
            queries = self.queries[dataset_name]
            scores = {}

            # Process queries in parallel
            with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
                # Submit all tasks
                future_to_query = {
                    executor.submit(
                        self.process_single_query,
                        query,
                        query_id,
                        top_k,
                        bm25_model,
                        doc_ids,
                    ): query_id
                    for query_id, query in queries.items()
                }

                # Collect results with progress bar
                for future in tqdm(
                    as_completed(future_to_query),
                    total=len(future_to_query),
                    desc="Processing queries",
                ):
                    query_id, results = future.result()
                    scores[query_id] = results

            self.scores[dataset_name] = scores

        return self.scores


class TFIDF:
    """TF-IDF search implementation with caching and parallel processing."""

    def __init__(
        self,
        corpus,
        dataset_type="hotpotqa",
        cache_dir="cache",
        use_cache=True,
        n_workers=4,
    ):
        self.original_corpus = corpus
        self.dataset_type = dataset_type
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.n_workers = n_workers
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.corpus_docs = []
        self.doc_ids = []
        self.setup()

    def setup(self):
        """Setup corpus and document IDs."""
        self.hash_id_to_text = {}
        self.text_to_hash_id = {}

        # Reset corpus_docs for building the searchable documents
        self.corpus_docs = []
        self.doc_ids = []

        # Build corpus and mappings efficiently
        if self.dataset_type in ["hotpotqa", "2wikimultihopqa"]:
            for item in self.original_corpus:
                ctxs = item["context"]
                for ctx in ctxs:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        self.hash_id_to_text[(title, idx)] = ct
                        self.text_to_hash_id[ct] = (title, idx)
                        self.corpus_docs.append(ct)
                        self.doc_ids.append((title, idx))
        elif self.dataset_type == "musique":
            for item in self.original_corpus:
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    self.hash_id_to_text[(title, idx)] = text
                    self.text_to_hash_id[text] = (title, idx)
                    self.corpus_docs.append(text)
                    self.doc_ids.append((title, idx))

        print(f"Built corpus with {len(self.corpus_docs)} unique documents")

    def _get_cache_path(self):
        """Generate cache file path based on corpus hash."""
        if not self.use_cache:
            return None

        corpus_str = "".join(self.corpus_docs[:100])
        corpus_hash = hashlib.md5(corpus_str.encode()).hexdigest()[:8]
        cache_path = os.path.join(
            self.cache_dir, f"tfidf_{self.dataset_type}_{corpus_hash}.pkl"
        )

        os.makedirs(self.cache_dir, exist_ok=True)
        return cache_path

    def _build_or_load_tfidf(self):
        """Build TF-IDF model or load from cache."""
        if self.tfidf_vectorizer is not None:
            return

        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading TF-IDF model from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    data = pickle.load(f)
                    self.tfidf_vectorizer = data["vectorizer"]
                    self.tfidf_matrix = data["matrix"]
                return
            except Exception as e:
                print(f"Failed to load from cache: {e}")

        # Build new TF-IDF model
        print("Building TF-IDF index...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=50000,  # Limit vocabulary size for memory efficiency
            stop_words="english",
            ngram_range=(1, 2),  # Use unigrams and bigrams
            min_df=2,  # Ignore terms that appear in less than 2 documents
            max_df=0.95,  # Ignore terms that appear in more than 95% of documents
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
            tqdm(self.corpus_docs, desc="Vectorizing documents")
        )

        # Save to cache
        if cache_path:
            print(f"Caching TF-IDF model to: {cache_path}")
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(
                        {
                            "vectorizer": self.tfidf_vectorizer,
                            "matrix": self.tfidf_matrix,
                        },
                        f,
                    )
            except Exception as e:
                print(f"Failed to save to cache: {e}")

    def _process_single_query(self, query, query_id, top_k):
        """Process a single query and return top-k results."""
        # Transform query to TF-IDF space
        query_vec = self.tfidf_vectorizer.transform([query])

        # Calculate cosine similarity
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # Use argpartition for O(n) selection
        if len(similarities) > top_k:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        else:
            top_indices = np.argsort(similarities)[::-1]

        return query_id, [
            (self.doc_ids[idx], float(similarities[idx])) for idx in top_indices
        ]

    def get_scores(self, top_k=10, questions=None):
        """Get TF-IDF scores with parallel processing."""
        self.scores = {}

        # Build or load TF-IDF model once
        self._build_or_load_tfidf()

        # Prepare queries and IDs
        queries = []
        query_ids = []
        query_data = questions if questions is not None else self.corpus

        for item in query_data:
            query = item["question"]
            query_id = item.get("_id", item.get("id"))
            queries.append(query)
            query_ids.append(query_id)

        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_query = {
                executor.submit(
                    self._process_single_query, query, query_id, top_k
                ): query_id
                for query, query_id in zip(queries, query_ids)
            }

            # Collect results with progress bar
            for future in tqdm(
                as_completed(future_to_query),
                total=len(future_to_query),
                desc="Processing queries",
            ):
                query_id, results = future.result()
                self.scores[query_id] = results

        print(f"Processed {len(self.scores)} queries")
        return self.scores


def load_dataset(dataset_path, benchmark="longembed"):
    """Load dataset based on format type.

    Args:
        dataset_path: Path to dataset file
        benchmark: Dataset format type
    """
    golds, queries, corpus = {}, {}, {}

    if benchmark == "longembed":
        for dataset_name in tqdm(os.listdir(dataset_path)):
            sub_dataset_path = os.path.join(dataset_path, dataset_name)

            golds[dataset_name], queries[dataset_name], corpus[dataset_name] = (
                {},
                {},
                {},
            )
            with open(os.path.join(sub_dataset_path, "qrels.json"), "r") as f:
                qrels = json.load(f)

                for item in qrels:
                    golds[dataset_name][item["qid"]] = item["doc_id"]

            with open(os.path.join(sub_dataset_path, "queries.json"), "r") as f:
                questions = json.load(f)

                for item in questions:
                    queries[dataset_name][item["qid"]] = item["text"]

            with open(os.path.join(sub_dataset_path, "corpus.json"), "r") as f:
                dataset_data = json.load(f)

                for item in dataset_data:
                    corpus[dataset_name][item["doc_id"]] = item["text"]
    elif benchmark == "litsearch":
        pass
    else:
        pass

    return golds, queries, corpus


def evaluate(gold, pred, benchmark="hotpotqa"):
    """Evaluate predictions against gold labels.

    Args:
        gold: Gold reference - for longembed: str (doc_id), for hotpotqa: list of tuples
        pred: Predictions - list of doc_ids for longembed, list of tuples for hotpotqa
        benchmark: Dataset type ("longembed", "hotpotqa", etc.)

    Returns:
        For longembed: (recall@k, precision@k, accuracy, ndcg@k)
        For hotpotqa: (precision, recall, f1)
    """
    if benchmark == "longembed":
        # For longembed, gold is a single doc_id string
        gold_doc_id = gold if isinstance(gold, str) else gold[0]
        pred_doc_ids = pred if isinstance(pred, list) else [pred]

        k = len(pred_doc_ids) if pred_doc_ids else 1

        # Recall@k: 1 if gold doc_id is in predicted list, else 0
        hit = 1 if gold_doc_id in pred_doc_ids else 0
        recall_at_k = hit

        # Precision@k: hit/k (since we retrieved k docs and only 1 is relevant)
        precision_at_k = hit / k

        # Accuracy: same as recall for single relevant document case
        accuracy = hit

        # NDCG@k
        if gold_doc_id in pred_doc_ids:
            # Find rank of gold document (1-indexed)
            rank = pred_doc_ids.index(gold_doc_id) + 1
            # DCG: 1/log2(rank+1) since relevance is binary (1 for relevant)
            dcg = 1.0 / np.log2(rank + 1)
            # IDCG: 1.0 (perfect ranking would have relevant doc at rank 1)
            idcg = 1.0
            ndcg_at_k = dcg / idcg
        else:
            ndcg_at_k = 0.0

        return precision_at_k, recall_at_k, accuracy, ndcg_at_k
    else:
        pass


def calculate_metrics(golds, preds, benchmark="hotpotqa"):
    """Calculate metrics per dataset.

    Args:
        golds: For longembed: Dict mapping dataset_name to {query_id: gold_labels}
               For hotpotqa: Dict mapping query_id to gold_labels
        preds: For longembed: Dict mapping dataset_name to {query_id: predictions}
               For hotpotqa: Dict mapping query_id to predictions
        benchmark: Dataset type ("longembed", "hotpotqa", etc.)

    Returns:
        For longembed: dict of {dataset_name: metrics_dict}
        For hotpotqa: (precision, recall, f1)
    """
    dataset_metrics = {}
    if benchmark == "longembed":
        for dataset_name in golds.keys():
            dataset_golds = golds[dataset_name]
            dataset_preds = preds.get(dataset_name, {})
            ds_prec, ds_recall, ds_acc, ds_ndcg = 0.0, 0.0, 0.0, 0.0
            ds_count = 0

            for query_id in dataset_golds.keys():
                if query_id in dataset_preds:
                    prec, recall, acc, ndcg = evaluate(
                        dataset_golds[query_id],
                        dataset_preds[query_id],
                        benchmark=benchmark,
                    )
                    ds_prec += prec
                    ds_recall += recall
                    ds_acc += acc
                    ds_ndcg += ndcg
                    ds_count += 1

            if ds_count > 0:
                ds_f1 = (
                    2 * (ds_prec / ds_count) * (ds_recall / ds_count)
                    / ((ds_prec / ds_count) + (ds_recall / ds_count))
                    if (ds_prec / ds_count) + (ds_recall / ds_count) > 0
                    else 0.0
                )
                dataset_metrics[dataset_name] = {
                    "precision": ds_prec / ds_count,
                    "recall": ds_recall / ds_count,
                    "f1": ds_f1,
                    "accuracy": ds_acc / ds_count,
                    "ndcg": ds_ndcg / ds_count,
                    "count": ds_count,
                }

    return dataset_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--search_method",
        type=str,
        default="bm25",
        choices=["bm25", "tfidf"],
        help="Search method to use",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="longembed",
        choices=["longembed", "litsearch", "mldr"],
        help="Dataset format type",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/LongEmbed",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="cache", help="Directory for caching models"
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of worker threads"
    )
    args = parser.parse_args()

    golds, queries, corpus = load_dataset(args.dataset_path, args.benchmark)

    if args.search_method == "bm25":
        print(f"Using BM25 search with {args.n_workers} workers")
        searcher = BM25(
            queries,
            corpus,
            args.benchmark,
            cache_dir=args.cache_dir,
            n_workers=args.n_workers,
        )
    elif args.search_method == "tfidf":
        print(f"Using TF-IDF search with {args.n_workers} workers")
        searcher = TFIDF(
            corpus,
            args.benchmark,
            cache_dir=args.cache_dir,
            n_workers=args.n_workers,
        )

    # Get scores
    top_k_scores = searcher.get_scores(top_k=args.top_k, questions=queries)
    # Keep nested structure: {dataset_name: {query_id: [doc_ids]}}
    preds = {}
    for dataset_name, dataset_scores in top_k_scores.items():
        preds[dataset_name] = {}
        for query_id, results in dataset_scores.items():
            preds[dataset_name][query_id] = [v[0] for v in results]

    # Save predictions
    os.makedirs("outputs", exist_ok=True)
    output_file = f"outputs/{args.search_method}_predictions.json"
    with open(output_file, "w") as f:
        json.dump(preds, f)

    metrics = calculate_metrics(golds, preds, benchmark=args.benchmark)
    print(f"\nBenchmark: {args.benchmark}")
    print(f"Search Method: {args.search_method.upper()}")

    if isinstance(metrics, dict):
        # Multi-dataset format: dict of {dataset_name: metrics}
        print(f"\nPer-Dataset Metrics (k={args.top_k}):")
        print(f"{'Dataset':<20} {'P@k':<8} {'R@k':<8} {'F1@k':<8} {'Acc':<8} {'NDCG@k':<8} {'Count'}")
        print("-" * 75)
        for ds_name in sorted(metrics.keys()):
            m = metrics[ds_name]
            print(
                f"{ds_name:<20} {m['precision']:<8.4f} {m['recall']:<8.4f} "
                f"{m['f1']:<8.4f} {m['accuracy']:<8.4f} {m['ndcg']:<8.4f} {m['count']}"
            )
    else:
        # Single dataset format: (precision, recall, f1)
        prec, recall, f1 = metrics
        print(f"Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    main()
