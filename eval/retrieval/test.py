import argparse
import json
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from kv_store import KVStore


class BM25(KVStore):
    def __init__(
        self,
        queries,
        corpus,
        benchmark="longembed",
        n_workers=4,
    ):
        self.queries = queries
        self.corpus = corpus
        self.benchmark = benchmark
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


def load_dataset(dataset_path, benchmark="longembed"):
    """Load dataset based on format type.

    Args:
        dataset_path: Path to dataset directory
        benchmark: Dataset format type ("longembed", "litsearch", "mldr")
        
    Returns:
        Tuple of (golds, queries, corpus) where each is:
        - golds: {dataset_name: {query_id: gold_doc_id or [gold_doc_ids]}}
        - queries: {dataset_name: {query_id: query_text}}
        - corpus: {dataset_name: {doc_id: doc_text}}
    """
    golds, queries, corpus = {}, {}, {}

    if benchmark == "longembed":
        # LongEmbed format: subdirectories with qrels.json, queries.json, corpus.json
        for dataset_name in tqdm(os.listdir(dataset_path)):
            sub_dataset_path = os.path.join(dataset_path, dataset_name)
            
            # Skip if not a directory
            if not os.path.isdir(sub_dataset_path):
                continue

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
        # LitSearch format:
        # - corpus.json: [{"corpusid": int, "title": str, "abstract": str, "full_paper": str}, ...]
        # - queries.json: [{"query": str, "corpusids": [int, ...]}, ...]
        # Relevance is embedded in queries via "corpusids"
        dataset_name = "litsearch"
        golds[dataset_name], queries[dataset_name], corpus[dataset_name] = {}, {}, {}
        
        # Load corpus
        with open(os.path.join(dataset_path, "corpus.json"), "r") as f:
            corpus_data = json.load(f)
            for item in corpus_data:
                doc_id = str(item["corpusid"])  # Convert to string for consistency
                title = item.get("title", "")
                abstract = item.get("abstract", "")
                # Combine title and abstract for retrieval (full_paper may be too long)
                full_text = f"{title} {abstract}".strip()
                corpus[dataset_name][doc_id] = abstract
        
        # Load queries - relevance is embedded via "corpusids"
        with open(os.path.join(dataset_path, "queries.json"), "r") as f:
            queries_data = json.load(f)
            for idx, item in enumerate(queries_data):
                query_id = str(idx)  # Use index as query_id since no explicit id
                queries[dataset_name][query_id] = item["query"]
                
                # Gold documents from corpusids
                corpusids = item.get("corpusids", [])
                if len(corpusids) == 1:
                    golds[dataset_name][query_id] = str(corpusids[0])
                elif len(corpusids) > 1:
                    golds[dataset_name][query_id] = [str(cid) for cid in corpusids]
                            
    elif benchmark == "mldr":
        # MLDR format:
        # - corpus.json: [{"docid": str, "text": str}, ...]
        # - test.json: [{"query_id": str, "query": str, "positive_passages": [{"docid": str, "text": str}, ...]}, ...]
        # Relevance is embedded in test.json via "positive_passages"
        dataset_name = "mldr"
        golds[dataset_name], queries[dataset_name], corpus[dataset_name] = {}, {}, {}
        
        # Load corpus
        corpus_file = os.path.join(dataset_path, "corpus.json")
        if os.path.exists(corpus_file):
            with open(corpus_file, "r") as f:
                corpus_data = json.load(f)
                for item in corpus_data:
                    if "manual" not in item:
                        continue
                    doc_id = item["docid"]
                    text = item.get("text", "")
                    corpus[dataset_name][doc_id] = text

        # Load queries and golds from test.json
        test_file = os.path.join(dataset_path, "test.json")
        if os.path.exists(test_file):
            with open(test_file, "r") as f:
                test_data = json.load(f)
                for item in test_data:
                    query_id = item["query_id"]
                    queries[dataset_name][query_id] = item["query"]
                    
                    # Gold documents from positive_passages
                    positive_passages = item.get("positive_passages", [])
                    relevant_docs = [p["docid"] for p in positive_passages]
                    if len(relevant_docs) == 1:
                        golds[dataset_name][query_id] = relevant_docs[0]
                    elif len(relevant_docs) > 1:
                        golds[dataset_name][query_id] = relevant_docs

    return golds, queries, corpus


def evaluate_retrieval(gold_doc_ids, pred_doc_ids, cutoffs=[1, 3, 5, 10, 20, 50, 100]):
    """
    Compute retrieval metrics for single or multiple relevant documents.
    
    Args:
        gold_doc_ids: Gold document ID(s) - single string or list of strings
        pred_doc_ids: List of predicted doc IDs, ranked by relevance
        cutoffs: List of k values for computing metrics at various cutoffs
    
    Returns:
        dict with all metrics
    """
    pred_doc_ids = pred_doc_ids if pred_doc_ids else []
    
    # Normalize gold_doc_ids to a set
    if isinstance(gold_doc_ids, str):
        gold_set = {gold_doc_ids}
    else:
        gold_set = set(gold_doc_ids)
    
    num_relevant = len(gold_set)
    
    # Find ranks of all relevant documents in predictions
    relevant_ranks = []
    for i, doc_id in enumerate(pred_doc_ids):
        if doc_id in gold_set:
            relevant_ranks.append(i + 1)  # 1-indexed
    
    # Metrics at various cutoffs
    recall_at = {}
    precision_at = {}
    ndcg_at = {}
    
    for cutoff in cutoffs:
        # Count relevant docs in top-cutoff
        relevant_in_topk = sum(1 for r in relevant_ranks if r <= cutoff)
        
        # Recall@cutoff: fraction of relevant docs found in top-cutoff
        recall_at[cutoff] = relevant_in_topk / num_relevant
        
        # Precision@cutoff: relevant found / cutoff
        precision_at[cutoff] = relevant_in_topk / cutoff
        
        # NDCG@cutoff
        # DCG = sum of 1/log2(rank+1) for each relevant doc found
        dcg = sum(1.0 / np.log2(r + 1) for r in relevant_ranks if r <= cutoff)
        # IDCG = sum of 1/log2(i+1) for i in 1..min(num_relevant, cutoff)
        idcg = sum(1.0 / np.log2(i + 1) for i in range(1, min(num_relevant, cutoff) + 1))
        ndcg_at[cutoff] = dcg / idcg if idcg > 0 else 0.0
    
    # MRR (Mean Reciprocal Rank): 1/rank of first relevant doc
    mrr = 1.0 / relevant_ranks[0] if relevant_ranks else 0.0
    
    return {
        'recall_at': recall_at,
        'precision_at': precision_at,
        'ndcg_at': ndcg_at,
        'mrr': mrr,
        'first_rank': relevant_ranks[0] if relevant_ranks else None,
    }


def calculate_metrics(golds, preds, benchmark="longembed", cutoffs=[1, 3, 5, 10, 20, 50, 100]):
    """Calculate metrics per dataset with various cutoffs.

    Args:
        golds: Dict mapping dataset_name to {query_id: gold_doc_id or [gold_doc_ids]}
        preds: Dict mapping dataset_name to {query_id: [pred_doc_ids]}
        benchmark: Dataset type ("longembed", "litsearch", "mldr")
        cutoffs: List of k values for computing metrics at various cutoffs

    Returns:
        dict of {dataset_name: metrics_dict}
    """
    dataset_metrics = {}
    
    # All benchmarks use the same evaluation logic now
    for dataset_name in golds.keys():
        dataset_golds = golds[dataset_name]
        dataset_preds = preds.get(dataset_name, {})
        
        # Initialize accumulators for each cutoff
        recall_sum = {k: 0.0 for k in cutoffs}
        precision_sum = {k: 0.0 for k in cutoffs}
        ndcg_sum = {k: 0.0 for k in cutoffs}
        mrr_sum = 0.0
        count = 0

        for query_id in dataset_golds.keys():
            if query_id in dataset_preds:
                metrics = evaluate_retrieval(
                    dataset_golds[query_id],
                    dataset_preds[query_id],
                    cutoffs=cutoffs,
                )
                
                for k in cutoffs:
                    recall_sum[k] += metrics['recall_at'][k]
                    precision_sum[k] += metrics['precision_at'][k]
                    ndcg_sum[k] += metrics['ndcg_at'][k]

                mrr_sum += metrics['mrr']
                count += 1

        if count > 0:
            dataset_metrics[dataset_name] = {
                'recall_at': {k: recall_sum[k] / count for k in cutoffs},
                'precision_at': {k: precision_sum[k] / count for k in cutoffs},
                'ndcg_at': {k: ndcg_sum[k] / count for k in cutoffs},
                'mrr': mrr_sum / count,
                'count': count,
            }

    return dataset_metrics


def print_metrics_table(dataset_metrics, cutoffs=[1, 3, 5, 10, 20, 50, 100]):
    """Print metrics in a formatted table."""
    
    for dataset_name, metrics in sorted(dataset_metrics.items()):
        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name} (n={metrics['count']})")
        print(f"{'=' * 80}")
        
        # Filter cutoffs to only those present in metrics
        available_cutoffs = [k for k in cutoffs if k in metrics['recall_at']]
        
        # Print header
        header = f"{'Metric':<12}"
        for k in available_cutoffs:
            header += f"@{k:<7}"
        print(header)
        print("-" * 80)
        
        # Print each metric row
        for metric_name in ['recall_at', 'precision_at', 'ndcg_at']:
            row = f"{metric_name.replace('_at', ''):<12}"
            for k in available_cutoffs:
                row += f"{metrics[metric_name][k]:<8.4f}"
            print(row)
        
        print("-" * 80)
        print(f"MRR: {metrics['mrr']:.4f}")


def print_summary_table(dataset_metrics, cutoffs=[1, 3, 5, 10, 20, 50, 100]):
    """Print a summary table comparing all datasets."""
    
    print(f"\n{'=' * 100}")
    print("SUMMARY: Recall@k Across Datasets")
    print(f"{'=' * 100}")
    
    # Filter cutoffs
    sample_metrics = next(iter(dataset_metrics.values()))
    available_cutoffs = [k for k in cutoffs if k in sample_metrics['recall_at']]
    
    # Header
    header = f"{'Dataset':<25}"
    for k in available_cutoffs:
        header += f"R@{k:<6}"
    header += f"{'MRR':<8}"
    print(header)
    print("-" * 100)
    
    # Rows for each dataset
    for dataset_name in sorted(dataset_metrics.keys()):
        m = dataset_metrics[dataset_name]
        row = f"{dataset_name:<25}"
        for k in available_cutoffs:
            row += f"{m['recall_at'][k]:<8.4f}"
        row += f"{m['mrr']:<8.4f}"
        print(row)
    
    # Average row
    print("-" * 100)
    avg_row = f"{'AVERAGE':<25}"
    for k in available_cutoffs:
        avg_recall = np.mean([m['recall_at'][k] for m in dataset_metrics.values()])
        avg_row += f"{avg_recall:<8.4f}"
    avg_mrr = np.mean([m['mrr'] for m in dataset_metrics.values()])
    avg_row += f"{avg_mrr:<8.4f}"
    print(avg_row)
    print(f"{'=' * 100}")


def main():
    parser = argparse.ArgumentParser()
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
        "--top_k", type=int, default=100, help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--cutoffs",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20, 50, 100],
        help="Cutoffs for evaluation metrics",
    )
    parser.add_argument(
        "--n_workers", type=int, default=4, help="Number of worker threads"
    )
    args = parser.parse_args()

    # Ensure top_k >= max cutoff
    max_cutoff = max(args.cutoffs)
    if args.top_k < max_cutoff:
        print(f"Warning: top_k ({args.top_k}) < max cutoff ({max_cutoff}). Setting top_k = {max_cutoff}")
        args.top_k = max_cutoff

    golds, queries, corpus = load_dataset(args.dataset_path, args.benchmark)

    print(f"Using BM25 search with {args.n_workers} workers")
    searcher = BM25(
        queries,
        corpus,
        args.benchmark,
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
    output_file = f"outputs/{args.benchmark}_bm25_predictions.json"
    with open(output_file, "w") as f:
        json.dump(preds, f)

    # Calculate and display metrics
    metrics = calculate_metrics(golds, preds, benchmark=args.benchmark, cutoffs=args.cutoffs)
    
    print(f"\n{'#' * 100}")
    print(f"Benchmark: {args.benchmark}")
    print("Search Method: BM25")
    print(f"Top-k: {args.top_k}")
    print(f"Cutoffs: {args.cutoffs}")
    print(f"{'#' * 100}")

    # Print detailed metrics for each dataset
    print_metrics_table(metrics, cutoffs=args.cutoffs)
    
    # Save metrics to file
    metrics_file = f"outputs/{args.benchmark}_bm25_metrics.json"
    with open(metrics_file, "w") as f:
        # Convert int keys to strings for JSON serialization
        serializable_metrics = {}
        for ds_name, ds_metrics in metrics.items():
            serializable_metrics[ds_name] = {
                'recall_at': {str(k): v for k, v in ds_metrics['recall_at'].items()},
                'precision_at': {str(k): v for k, v in ds_metrics['precision_at'].items()},
                'ndcg_at': {str(k): v for k, v in ds_metrics['ndcg_at'].items()},
                'mrr': ds_metrics['mrr'],
                'count': ds_metrics['count'],
            }
        json.dump(serializable_metrics, f, indent=2)
    
    print(f"\nPredictions saved to: {output_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()