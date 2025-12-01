import json
import argparse
from rank_bm25 import BM25Okapi

class BM25:
    def __init__(self, dataset):
        self.dataset = dataset
        self.setup()

    def setup(self):
        self.hash_id_to_text = {}
        self.text_to_hash_id = {}
        for item in self.dataset:
            for ctx in item['context']:
                title = ctx[0]
                for idx, ct in enumerate(ctx[1]):
                    self.hash_id_to_text[(title, idx)] = ct
                    self.text_to_hash_id[ct] = (title, idx)

    def get_scores(self, top_k=10):
        self.scores = {}
        for item in self.dataset:
            query = item["question"]
            corpus = []
            doc_ids = []  # Store (title, idx) for each document
            
            for ctx in item['context']:
                title = ctx[0]
                for idx, ct in enumerate(ctx[1]):
                    corpus.append(ct)
                    doc_ids.append((title, idx))
            
            # Tokenize corpus
            tokenized_corpus = [doc.split() for doc in corpus]
            bm25 = BM25Okapi(tokenized_corpus)
            
            # Tokenize query and get scores
            tokenized_query = query.split()
            scores_array = bm25.get_scores(tokenized_query)
            
            # Map each (title, idx) to its score
            self.scores[item['_id']] = {doc_id: score for doc_id, score in zip(doc_ids, scores_array)}
        print(f"Processed {len(self.scores)} queries")
        top_k_scores = {}
        for item in self.scores:
            top_k_scores[item] = sorted(self.scores[item].items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_k_scores
    
def load_dataset(dataset_path):
    with open(dataset_path, 'r') as f:
        return json.load(f)

def evaluate(golds, preds):
    tp, fp, fn = 0, 0, 0
    for key in preds.keys():
        cur_sp_pred = preds[key]
        gold_sp_pred = golds[key]
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
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='datasets/hotpotqa/hotpot_dev_fullwiki_v1_100.json')
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    golds = {item["_id"]: [(sps[0], sps[1]) for sps in item["supporting_facts"]] for item in dataset}
    bm25 = BM25(dataset)
    top_k_scores = bm25.get_scores(top_k=10)  # Then get top k
    preds = {key: [v[0] for v in value] for key, value in top_k_scores.items()}
    prec, recall, f1 = evaluate(golds, preds)
    print(f"Precision: {prec}, Recall: {recall}, F1: {f1}")

if __name__ == "__main__":
    main()