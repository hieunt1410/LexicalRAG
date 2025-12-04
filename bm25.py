import json
import argparse
from rank_bm25 import BM25Okapi


class BM25:
    def __init__(self, dataset, dataset_type="hotpotqa"):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.setup()

    def setup(self):
        self.hash_id_to_text = {}
        self.text_to_hash_id = {}

        if self.dataset_type in ["hotpotqa", "triviaqa"]:
            # HotpotQA and TriviaQA format: context is [title, [sentences]]
            for item in self.dataset:
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        self.hash_id_to_text[(title, idx)] = ct
                        self.text_to_hash_id[ct] = (title, idx)
        elif self.dataset_type == "musique":
            # MuSiQue format: paragraphs is list of {idx, title, paragraph_text}
            for item in self.dataset:
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    self.hash_id_to_text[(title, idx)] = text
                    self.text_to_hash_id[text] = (title, idx)

    def get_scores(self, top_k=10):
        self.scores = {}
        corpus = []
        doc_ids = []
        for item in self.dataset:
            query = item["question"]

            if self.dataset_type in ["hotpotqa", "triviaqa"]:
                # HotpotQA and TriviaQA format
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        corpus.append(ct)
                        doc_ids.append((title, idx))
            elif self.dataset_type == "musique":
                # MuSiQue format
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    corpus.append(text)
                    doc_ids.append((title, idx))

        tokenized_corpus = [doc.split() for doc in corpus]

        bm25 = BM25Okapi(tokenized_corpus)

        for item in self.dataset:
            query = item["question"]
            # Skip items with empty corpus
            if not tokenized_corpus:
                continue

            # Tokenize query and get scores
            tokenized_query = query.split()
            scores_array = bm25.get_scores(tokenized_query)

            # Map each (title, idx) to its score
            query_id = item.get("_id") or item.get(
                "id"
            )  # HotpotQA uses '_id', MuSiQue uses 'id'
            self.scores[query_id] = {
                doc_id: score for doc_id, score in zip(doc_ids, scores_array)
            }
        print(f"Processed {len(self.scores)} queries")
        top_k_scores = {}
        for item in self.scores:
            top_k_scores[item] = sorted(
                self.scores[item].items(), key=lambda x: x[1], reverse=True
            )[:top_k]
        return top_k_scores


def load_dataset(dataset_path, dataset_type="hotpotqa"):
    """Load dataset based on format type.

    Args:
        dataset_path: Path to dataset file
        dataset_type: 'hotpotqa'/'triviaqa' for JSON format or 'musique' for JSONL format
    """
    with open(dataset_path, "r") as f:
        if dataset_type in ["hotpotqa", "triviaqa"]:
            # HotpotQA and TriviaQA: Single JSON array
            return json.load(f)
        elif dataset_type == "musique":
            # MuSiQue: JSONL format (one JSON object per line)
            return [json.loads(line) for line in f]
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")


def equals(a, b):
    if a[0] in b[0] or b[0] in a[0]:
        if a[1] == b[1]:
            return True
    return False


def evaluate(gold, pred):
    tp, fp, fn = 0, 0, 0

    for pred_elem in pred:
        matched = False
        for gold_elem in gold:
            if equals(pred_elem, gold_elem):
                matched = True
                break  # Found a match, stop searching

        if matched:
            tp += 1
        else:
            fp += 1

    # Count FN
    for gold_elem in gold:
        matched = False
        for pred_elem in pred:
            if equals(gold_elem, pred_elem):
                matched = True
                break  # Found a match, stop searching

        if not matched:
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
        "--dataset_type",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "musique", "triviaqa"],
        help="Dataset format type",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/hotpotqa/hotpot.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--top_k", type=int, default=10, help="Number of documents to retrieve"
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path, args.dataset_type)

    # Extract gold supporting facts based on dataset type
    if args.dataset_type in ["hotpotqa", "triviaqa"]:
        golds = {
            item["_id"]: [(sps[0], sps[1]) for sps in item["supporting_facts"]]
            for item in dataset
        }
    elif args.dataset_type == "musique":
        golds = {
            item["id"]: [
                (para["title"], para["idx"])
                for para in item["paragraphs"]
                if para["is_supporting"]
            ]
            for item in dataset
        }

    bm25 = BM25(dataset, args.dataset_type)
    top_k_scores = bm25.get_scores(top_k=args.top_k)
    preds = {key: [v[0] for v in value] for key, value in top_k_scores.items()}
    prec, recall, f1 = calculate_metrics(golds, preds)
    print(f"Dataset: {args.dataset_type}")
    print(f"Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")


if __name__ == "__main__":
    main()
