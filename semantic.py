import argparse
import torch
from typing import List
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagModel

from bm25 import load_dataset, evaluate
from config.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE
from config.config import RERANKER_MODEL

class BaseRAG:

    def __init__(self, dataset):
        self.dataset = dataset
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker_model = FlagModel(RERANKER_MODEL)
        
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

    def encoder_single(self, text: str) -> torch.Tensor:
        return self.embedding_model.encode(text, convert_to_tensor=True)
    
    def encode_batch(
        self, texts: List[str], batch_size: int = EMBEDDING_BATCH_SIZE
    ) -> torch.Tensor:
        """Encode texts in batches."""
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = self.embedding_model.encode(batch, convert_to_tensor=True)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def get_scores(self, top_k=10):
        self.scores = {}
        for item in self.dataset:
            query_embedding = self.encoder_single(item["question"])
            corpus = []
            doc_ids = []  # Store (title, idx) for each document

            for ctx in item["context"]:
                title = ctx[0]
                for idx, ct in enumerate(ctx[1]):
                    corpus.append(ct)
                    doc_ids.append((title, idx))

            corpus_embeddings = self.encode_batch(corpus)
            # Normalize embeddings for cosine similarity
            query_embedding = query_embedding / query_embedding.norm()
            corpus_embeddings = corpus_embeddings / corpus_embeddings.norm(dim=1, keepdim=True)
            scores = torch.matmul(query_embedding, corpus_embeddings.T)
            self.scores[item["_id"]] = {
                doc_id: score.item() for doc_id, score in zip(doc_ids, scores)
            }
        print(f"Processed {len(self.scores)} queries")
        
        top_k_scores = {}
        for item in self.scores:
            top_k_scores[item] = sorted(self.scores[item].items(), key=lambda x: x[1], reverse=True)[:top_k]
        return top_k_scores

    def rerank(self, top_k_scores):
        reranked_scores = {}
        for item in self.dataset:
            id = item["_id"]
            query = item["question"]
            doc_ids = []
            pairs = []
            for doc_id, _ in top_k_scores[id]:
                pairs.append([query, self.hash_id_to_text[doc_id]])
                doc_ids.append(doc_id)
            rerank_scores = self.reranker_model.compute_score(pairs)
            reranked_scores[id] = {doc_id: score for doc_id, score in zip(doc_ids, rerank_scores)}
        return reranked_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="datasets/hotpotqa/hotpot_dev_fullwiki_v1_100.json",
    )
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    golds = {item["_id"]: [(sps[0], sps[1]) for sps in item["supporting_facts"]] for item in dataset}
    base_rag = BaseRAG(dataset)
    top_k_scores = base_rag.get_scores(top_k=10)
    preds = {key: [v[0] for v in value] for key, value in top_k_scores.items()}
    prec, recall, f1 = evaluate(golds, preds)
    print("Metrics before rerank:")
    print(f"Precision: {prec}, Recall: {recall}, F1: {f1}")
    print("--------------------------------")
    reranked_scores = base_rag.rerank(top_k_scores)
    # Combine initial retrieval scores with reranker scores
    final_scores = {}
    for key in top_k_scores.keys():
        combined_scores = {}
        # Create dict from top_k_scores for easy lookup
        top_k_dict = {doc_id: score for doc_id, score in top_k_scores[key]}
        
        # Combine scores: initial_score + reranker_score
        for doc_id in reranked_scores[key].keys():
            initial_score = top_k_dict[doc_id]
            rerank_score = reranked_scores[key][doc_id]
            combined_scores[doc_id] = initial_score + rerank_score
        
        # Sort by combined scores
        final_scores[key] = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Extract just the document IDs for evaluation
    preds = {key: [v[0] for v in value] for key, value in final_scores.items()}
    prec, recall, f1 = evaluate(golds, preds)
    print("Metrics after rerank:")
    print(f"Precision: {prec}, Recall: {recall}, F1: {f1}")
    print("--------------------------------")

if __name__ == "__main__":
    main()
