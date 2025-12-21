import argparse
import json
import re
import os
import pickle
import hashlib
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from rank_bm25 import BM25Okapi
from tqdm import tqdm


def normalize_entity_name(entity):
    """
    Remove disambiguators from entity names.
    Examples:
        "Ed Wood (film)" -> "Ed Wood"
        "Deliver Us from Evil (2014 film)" -> "Deliver Us from Evil"
        "Scott Derrickson" -> "Scott Derrickson" (unchanged)
    """
    # Remove parenthetical disambiguators at the end
    entity = re.sub(r"\s*\([^)]*\)\s*$", "", entity)
    return entity.strip()


class BM25:
    def __init__(
        self,
        questions,
        corpus,
        dataset_type="hotpotqa",
        cache_dir="cache",
        use_cache=True,
        n_workers=4,
    ):
        self.questions = questions
        self.corpus = corpus
        self.dataset_type = dataset_type
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.n_workers = n_workers
        self.bm25_model = None
        self.doc_ids = []
        self.setup()

    def setup(self):
        self.hash_id_to_text = {}
        self.text_to_hash_id = {}

        # Build corpus and mappings efficiently
        if self.dataset_type in ["hotpotqa", "2wikimultihopqa"]:
            # HotpotQA format: context is [title, [sentences]]
            for item in self.corpus:
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        self.hash_id_to_text[(title, idx)] = ct
                        self.text_to_hash_id[ct] = (title, idx)
                        self.corpus.append(ct)
                        self.doc_ids.append((title, idx))
        elif self.dataset_type == "musique":
            # MuSiQue format: paragraphs is list of {idx, title, paragraph_text}
            for item in self.corpus:
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    self.hash_id_to_text[(title, idx)] = text
                    self.text_to_hash_id[text] = (title, idx)
                    self.corpus.append(text)
                    self.doc_ids.append((title, idx))

        print(f"Built corpus with {len(self.corpus)} unique documents")

    def _get_cache_path(self):
        """Generate cache file path based on corpus hash."""
        if not self.use_cache:
            return None

        # Create hash from corpus content for cache key
        corpus_str = "".join(self.corpus[:100])  # Use first 100 docs for hashing
        corpus_hash = hashlib.md5(corpus_str.encode()).hexdigest()[:8]
        cache_path = os.path.join(
            self.cache_dir, f"bm25_{self.dataset_type}_{corpus_hash}.pkl"
        )

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)
        return cache_path

    def _build_or_load_bm25(self):
        """Build BM25 model or load from cache."""
        if self.bm25_model is not None:
            return

        cache_path = self._get_cache_path()

        # Try to load from cache
        if cache_path and os.path.exists(cache_path):
            print(f"Loading BM25 model from cache: {cache_path}")
            try:
                with open(cache_path, "rb") as f:
                    self.bm25_model = pickle.load(f)
                return
            except Exception as e:
                print(f"Failed to load from cache: {e}")

        # Build new BM25 model
        print("Building BM25 index...")
        tokenized_corpus = [doc.split() for doc in tqdm(self.corpus)]
        self.bm25_model = BM25Okapi(tokenized_corpus)

        # Save to cache
        if cache_path:
            print(f"Caching BM25 model to: {cache_path}")
            try:
                with open(cache_path, "wb") as f:
                    pickle.dump(self.bm25_model, f)
            except Exception as e:
                print(f"Failed to save to cache: {e}")

    def _process_single_query(self, query, query_id, top_k):
        """Process a single query and return top-k results."""
        tokenized_query = query.split()
        scores_array = self.bm25_model.get_scores(tokenized_query)

        # Use argpartition for O(n) selection instead of O(n log n) sort
        if len(scores_array) > top_k:
            # Get indices of top-k scores efficiently
            top_indices = np.argpartition(scores_array, -top_k)[-top_k:]
            # Sort only the top-k results
            top_indices = top_indices[np.argsort(scores_array[top_indices])[::-1]]
        else:
            top_indices = np.argsort(scores_array)[::-1]

        return query_id, [
            (self.doc_ids[idx], float(scores_array[idx])) for idx in top_indices
        ]

    def get_scores(self, top_k=10, questions=None):
        """Get BM25 scores with parallel processing."""
        self.scores = {}

        # Build or load BM25 model once
        self._build_or_load_bm25()

        # Prepare queries and IDs
        queries = []
        query_ids = []
        query_data = questions if questions else self.questions

        for item in query_data:
            query = item["question"]
            query_id = item.get("_id", item.get("id"))
            queries.append(query)
            query_ids.append(query_id)

        # Process queries in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all tasks
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
        self.corpus = corpus
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

        # Build corpus and mappings efficiently
        if self.dataset_type in ["hotpotqa", "2wikimultihopqa"]:
            for item in self.corpus:
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        self.hash_id_to_text[(title, idx)] = ct
                        self.text_to_hash_id[ct] = (title, idx)
                        self.corpus_docs.append(ct)
                        self.doc_ids.append((title, idx))
        elif self.dataset_type == "musique":
            for item in self.corpus:
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


def load_dataset(dataset_path, dataset_type="hotpotqa"):
    """Load dataset based on format type.

    Args:
        dataset_path: Path to dataset file
        dataset_type: 'hotpotqa'/'triviaqa' for JSON format or 'musique' for JSONL format
    """
    questions, corpus, facts = [], [], []
    with open(dataset_path + "/questions.json", "r") as f:
        questions = json.load(f)
    with open(dataset_path + "/corpus.json", "r") as f:
        corpus = json.load(f)
    with open(dataset_path + "/facts.json", "r") as f:
        facts = json.load(f)

    return questions, corpus, facts


def evaluate(gold, pred):
    cur_sp_pred = set(map(tuple, pred))
    gold_sp_pred = set(map(tuple, gold))
    gold_sp_pred = set(
        (normalize_entity_name(title), sent_id) for title, sent_id in gold_sp_pred
    )
    cur_sp_pred = set(
        (normalize_entity_name(title), sent_id) for title, sent_id in cur_sp_pred
    )

    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0

    return prec, recall, f1


def calculate_metrics(golds, preds):
    all_prec, all_recall, all_f1 = 0.0, 0.0, 0.0
    count = 0
    for cur_id in golds.keys():
        if cur_id in preds:
            prec, recall, f1 = evaluate(
                golds[cur_id],
                preds[cur_id],
            )
            all_prec += prec
            all_recall += recall
            all_f1 += f1
            count += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    return all_prec / count, all_recall / count, all_f1 / count


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
        "--dataset_type",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "musique", "triviaqa"],
        help="Dataset format type",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/hotpotqa",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--cache_dir", type=str, default="cache", help="Directory for caching models"
    )
    parser.add_argument("--no_cache", action="store_true", help="Disable caching")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of worker threads"
    )
    args = parser.parse_args()

    questions, corpus, facts = load_dataset(args.dataset_path, args.dataset_type)

    # Extract gold supporting facts based on dataset type
    if args.dataset_type in ["hotpotqa", "2wikimultihopqa"]:
        golds = {}
        for item in facts:
            golds[item["_id"]] = [
                (title, sent_id) for title, sent_id in item["supporting_facts"]
            ]

    elif args.dataset_type == "musique":
        golds = {
            item["id"]: [
                (para["title"], para["idx"])
                for para in item["paragraphs"]
                if para["is_supporting"]
            ]
            for item in facts
        }

    # Initialize searcher based on method
    use_cache = not args.no_cache
    if args.search_method == "bm25":
        print(f"Using BM25 search with {args.n_workers} workers")
        searcher = BM25(
            questions,
            corpus,
            args.dataset_type,
            cache_dir=args.cache_dir,
            use_cache=use_cache,
            n_workers=args.n_workers,
        )
    elif args.search_method == "tfidf":
        print(f"Using TF-IDF search with {args.n_workers} workers")
        searcher = TFIDF(
            corpus,
            args.dataset_type,
            cache_dir=args.cache_dir,
            use_cache=use_cache,
            n_workers=args.n_workers,
        )

    # Get scores
    top_k_scores = searcher.get_scores(top_k=args.top_k, questions=questions)
    preds = {key: [v[0] for v in value] for key, value in top_k_scores.items()}

    # Save predictions
    output_file = f"{args.search_method}_predictions.json"
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)

    prec, recall, f1 = calculate_metrics(golds, preds)
    print(f"\nDataset: {args.dataset_type}")
    print(f"Search Method: {args.search_method.upper()}")
    print(f"Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print(f"Predictions saved to: {output_file}")


if __name__ == "__main__":
    main()
