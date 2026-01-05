import argparse
import json
import torch
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from FlagEmbedding import FlagReranker

from bm25 import load_dataset, calculate_metrics
from config.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, RERANKER_MODEL


class BaseSearch(ABC):
    """Abstract base class for search strategies."""

    def __init__(self, dataset, dataset_type="hotpotqa"):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.hash_id_to_text = {}
        self.text_to_hash_id = {}
        self.setup()

    def setup(self):
        """Setup mapping between hash IDs and text content."""
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

    @abstractmethod
    def search(self, top_k: int = 10) -> Dict[str, List[Tuple[Tuple[str, int], float]]]:
        """Perform search and return scores."""
        pass


class EmbeddingOnlySearch(BaseSearch):
    """Search using only embedding-based cosine similarity."""

    def __init__(self, dataset, dataset_type="hotpotqa"):
        super().__init__(dataset, dataset_type)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    def encoder_single(self, text: str) -> torch.Tensor:
        """Encode a single text."""
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

    def search(self, top_k: int = 10) -> Dict[str, List[Tuple[Tuple[str, int], float]]]:
        """Perform embedding-based search."""
        self.scores = {}
        corpus = []
        doc_ids = []

        # Build corpus and document IDs
        for item in self.dataset:
            if self.dataset_type in ["hotpotqa", "triviaqa"]:
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        corpus.append(ct)
                        doc_ids.append((title, idx))
            elif self.dataset_type == "musique":
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    corpus.append(text)
                    doc_ids.append((title, idx))

        # Encode all corpus documents
        corpus_embeddings = self.encode_batch(corpus)
        corpus_embeddings = corpus_embeddings / corpus_embeddings.norm(
            dim=1, keepdim=True
        )

        # For each query, compute cosine similarity
        for item in self.dataset:
            query_embedding = self.encoder_single(item["question"])
            query_embedding = query_embedding / query_embedding.norm()

            scores = torch.matmul(query_embedding, corpus_embeddings.T)
            query_id = (
                item.get("_id")
                if self.dataset_type in ["hotpotqa", "triviaqa"]
                else item.get("id")
            )

            # Get top-k results
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            self.scores[query_id] = [
                (doc_ids[idx], score.item())
                for idx, score in zip(top_indices, top_scores)
            ]

        print(f"Processed {len(self.scores)} queries using embedding search")
        return self.scores


class RerankerOnlySearch(BaseSearch):
    """Search using only reranker (cross-encoder) on all documents."""

    def __init__(self, dataset, dataset_type="hotpotqa"):
        super().__init__(dataset, dataset_type)
        self.reranker_model = FlagReranker(RERANKER_MODEL, use_fp16=True)

    def search(self, top_k: int = 10) -> Dict[str, List[Tuple[Tuple[str, int], float]]]:
        """Perform reranker-only search on all documents."""
        self.scores = {}

        # Pre-compute all documents for each query
        for item in self.dataset:
            query_id = (
                item.get("_id")
                if self.dataset_type in ["hotpotqa", "triviaqa"]
                else item.get("id")
            )
            query = item["question"]

            # Get all documents
            all_docs = []
            doc_ids = []

            if self.dataset_type in ["hotpotqa", "triviaqa"]:
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        all_docs.append(ct)
                        doc_ids.append((title, idx))
            elif self.dataset_type == "musique":
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    all_docs.append(text)
                    doc_ids.append((title, idx))

            # Create query-document pairs
            pairs = [[query, doc] for doc in all_docs]

            # Compute reranker scores
            rerank_scores = self.reranker_model.compute_score(pairs)

            # Pair scores with document IDs and sort
            doc_scores = list(zip(doc_ids, rerank_scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)

            # Keep top-k
            self.scores[query_id] = doc_scores[:top_k]

        print(f"Processed {len(self.scores)} queries using reranker-only search")
        return self.scores


class CombinedSearch(BaseSearch):
    """Search using both embedding for initial retrieval and reranker for re-ranking."""

    def __init__(self, dataset, dataset_type="hotpotqa"):
        super().__init__(dataset, dataset_type)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker_model = FlagReranker(RERANKER_MODEL, use_fp16=True)

    def encoder_single(self, text: str) -> torch.Tensor:
        """Encode a single text."""
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

    def search(
        self,
        top_k: int = 10,
        initial_retrieval_k: int = 100,
        combine_scores: bool = True,
    ) -> Dict[str, List[Tuple[Tuple[str, int], float]]]:
        """
        Perform combined search using embedding for initial retrieval and reranker for re-ranking.

        Args:
            top_k: Final number of documents to return
            initial_retrieval_k: Number of documents to retrieve with embedding before reranking
            combine_scores: Whether to combine embedding and reranker scores (True) or use only reranker scores (False)
        """
        # First, perform embedding-based search to get initial candidates
        corpus = []
        doc_ids = []

        for item in self.dataset:
            if self.dataset_type in ["hotpotqa", "triviaqa"]:
                for ctx in item["context"]:
                    title = ctx[0]
                    for idx, ct in enumerate(ctx[1]):
                        corpus.append(ct)
                        doc_ids.append((title, idx))
            elif self.dataset_type == "musique":
                for para in item["paragraphs"]:
                    title = para["title"]
                    idx = para["idx"]
                    text = para["paragraph_text"]
                    corpus.append(text)
                    doc_ids.append((title, idx))

        # Encode corpus once
        corpus_embeddings = self.encode_batch(corpus)
        corpus_embeddings = corpus_embeddings / corpus_embeddings.norm(
            dim=1, keepdim=True
        )

        self.scores = {}

        for item in self.dataset:
            query_id = (
                item.get("_id")
                if self.dataset_type in ["hotpotqa", "triviaqa"]
                else item.get("id")
            )
            query = item["question"]

            # Initial embedding search
            query_embedding = self.encoder_single(query)
            query_embedding = query_embedding / query_embedding.norm()

            embedding_scores = torch.matmul(query_embedding, corpus_embeddings.T)

            # Get top-k initial candidates
            initial_k = min(initial_retrieval_k, len(embedding_scores))
            top_scores, top_indices = torch.topk(embedding_scores, initial_k)

            # Prepare for reranking
            candidate_docs = [corpus[idx] for idx in top_indices]
            candidate_doc_ids = [doc_ids[idx] for idx in top_indices]
            candidate_embedding_scores = [score.item() for score in top_scores]

            # Rerank the candidates
            pairs = [[query, doc] for doc in candidate_docs]
            rerank_scores = self.reranker_model.compute_score(pairs)

            # Combine scores if requested
            if combine_scores:
                # Normalize scores to combine them
                # Simple approach: use min-max scaling for both score types
                min_emb = min(candidate_embedding_scores)
                max_emb = max(candidate_embedding_scores)
                norm_emb_scores = [
                    (score - min_emb) / (max_emb - min_emb) if max_emb > min_emb else 0
                    for score in candidate_embedding_scores
                ]

                min_rerank = min(rerank_scores)
                max_rerank = max(rerank_scores)
                norm_rerank_scores = [
                    (score - min_rerank) / (max_rerank - min_rerank)
                    if max_rerank > min_rerank
                    else 0
                    for score in rerank_scores
                ]

                # Combine with equal weight
                combined_scores = [
                    (emb + rerank) / 2
                    for emb, rerank in zip(norm_emb_scores, norm_rerank_scores)
                ]

                doc_scores = list(zip(candidate_doc_ids, combined_scores))
            else:
                # Use only reranker scores
                doc_scores = list(zip(candidate_doc_ids, rerank_scores))

            # Sort and keep final top-k
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            self.scores[query_id] = doc_scores[:top_k]

        print(f"Processed {len(self.scores)} queries using combined search")
        return self.scores


def get_top_k_scores(scores, top_k=10):
    """Utility function to extract top-k scores from a scores dictionary."""
    return {
        item: sorted(scores[item].items(), key=lambda x: x[1], reverse=True)[:top_k]
        for item in scores
    }


def main():
    parser = argparse.ArgumentParser(
        description="Semantic Search with Different Strategies"
    )
    parser.add_argument(
        "--search_type",
        type=str,
        default="combined",
        choices=["embedding", "reranker", "combined"],
        help="Search strategy to use: embedding-only, reranker-only, or combined",
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
        default="datasets/hotpotqa/hotpotqa.json",
        help="Path to dataset file",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of documents to retrieve",
    )
    parser.add_argument(
        "--initial_retrieval_k",
        type=int,
        default=100,
        help="Number of documents for initial retrieval (used only with combined search)",
    )
    parser.add_argument(
        "--combine_scores",
        action="store_true",
        default=True,
        help="Combine embedding and reranker scores (used only with combined search)",
    )
    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(args.dataset_path, args.dataset_type)

    # Extract gold supporting facts based on dataset type
    if args.dataset_type in ["hotpotqa", "triviaqa"]:
        golds = {}
        for item in dataset:
            golds[item.get("_id", item.get("id"))] = [
                (title, sent_id) for title, sent_id in item["supporting_facts"]
            ]
    elif args.dataset_type == "musique":
        golds = {
            item["id"]: [
                (para["title"], para["idx"])
                for para in item["paragraphs"]
                if para["is_supporting"]
            ]
            for item in dataset
        }

    # Initialize and run the appropriate search strategy
    if args.search_type == "embedding":
        print("Using Embedding-Only Search")
        searcher = EmbeddingOnlySearch(dataset, args.dataset_type)
        scores = searcher.search(top_k=args.top_k)

    elif args.search_type == "reranker":
        print("Using Reranker-Only Search")
        searcher = RerankerOnlySearch(dataset, args.dataset_type)
        scores = searcher.search(top_k=args.top_k)

    elif args.search_type == "combined":
        print("Using Combined Embedding + Reranker Search")
        searcher = CombinedSearch(dataset, args.dataset_type)
        scores = searcher.search(
            top_k=args.top_k,
            initial_retrieval_k=args.initial_retrieval_k,
            combine_scores=args.combine_scores,
        )

    # Extract document IDs for evaluation
    preds = {key: [v[0] for v in value] for key, value in scores.items()}

    # Save predictions
    output_file = f"predictions_{args.search_type}.json"
    with open(output_file, "w") as f:
        json.dump(preds, f, indent=2, ensure_ascii=False)
    print(f"Predictions saved to {output_file}")

    # Calculate and display metrics
    prec, recall, f1 = calculate_metrics(golds, preds)
    print("\nMetrics:")
    print(f"Precision: {prec:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    print("--------------------------------")


if __name__ == "__main__":
    main()
