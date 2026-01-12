from utils import utils
import argparse
from typing import List


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str, required=True)
    parser.add_argument(
        "--topk",
        type=int,
        nargs="+",
        required=False,
        default=[1, 2, 3, 5, 10, 20, 50, 100],
    )
    return parser.parse_args()


def evaluate_metrics(predictions: List[dict], topk: List[int]) -> dict:
    """
    Evaluate retrieval metrics for each k value.
    
    For each query, we evaluate the top-k retrieved documents against
    the ground truth relevant documents (corpusids).
    """
    metrics = {}
    num_queries = len(predictions)
    
    for k in topk:
        prec_sum = 0
        recall_sum = 0
        f1_sum = 0
        ndcg_sum = 0
        
        for item in predictions:
            relevant_docs = item["corpusids"]
            retrieved_at_k = item["retrieved"][:k]

            prec = utils.calculate_precision(retrieved_at_k, relevant_docs)
            recall = utils.calculate_recall(retrieved_at_k, relevant_docs)
            f1 = 2 * (prec * recall) / (prec + recall) if (prec + recall) > 0 else 0
            ndcg = utils.calculate_ndcg(retrieved_at_k, relevant_docs)
            
            prec_sum += prec
            recall_sum += recall
            f1_sum += f1
            ndcg_sum += ndcg

        metrics[k] = {
            "prec": prec_sum / num_queries,
            "recall": recall_sum / num_queries,
            "f1": f1_sum / num_queries,
            "ndcg": ndcg_sum / num_queries,
        }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty print evaluation metrics in a formatted table."""
    header = f"{'k':>6} │ {'Prec':>8} │ {'Recall':>8} │ {'F1':>8} │ {'NDCG':>8}"
    separator = "─" * len(header)

    print("\n" + "═" * len(header))
    print("  RETRIEVAL EVALUATION RESULTS")
    print("═" * len(header))
    print(header)
    print(separator)

    for k, scores in sorted(metrics.items()):
        print(
            f"{k:>6} │ {scores['prec']:>8.4f} │ {scores['recall']:>8.4f} │ "
            f"{scores['f1']:>8.4f} │ {scores['ndcg']:>8.4f}"
        )

    print(separator)
    print()


if __name__ == "__main__":
    args = parse_args()
    predictions = utils.read_json(args.predictions_path)
    metrics = evaluate_metrics(predictions, topk=args.topk)
    print_metrics(metrics)
