import numpy as np
from typing import List, Dict, Optional
from FlagEmbedding import BGEM3FlagModel
from eval.retrieval.kv_store import KVStore, TextType


class BGE(KVStore):
    def __init__(
        self,
        index_name: str,
        model_path: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        max_length: int = 8192,
    ):
        super().__init__(index_name, "bge")
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.max_length = max_length
        self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16)

    def _encode_batch(  # noqa: ARG002
        self, texts: List[str], _type: TextType, show_progress_bar: bool = True
    ) -> np.ndarray:
        output = self._model.encode(
            texts,
            batch_size=32,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        embeddings = output["dense_vecs"].astype(np.float16)

        return embeddings

    def _query(self, encoded_query: np.ndarray, n: int) -> tuple[List[int], np.ndarray]:
        encoded_keys = np.array(self.encoded_keys)
        scores = np.dot(encoded_keys, encoded_query).astype(np.float16)
        top_indices = np.argsort(scores)[::-1][:n].tolist()
        return top_indices, scores[top_indices]

    def compute_pair_scores(
        self,
        text_pairs: List[List[str]],
        max_passage_length: Optional[int] = None,
        weights_for_different_modes: Optional[List[float]] = None,
    ) -> Dict[str, List[float]]:
        if max_passage_length is None:
            max_passage_length = self.max_length
        if weights_for_different_modes is None:
            weights_for_different_modes = [1.0, 0.0, 0.0]  # dense only by default

        return self._model.compute_score(
            text_pairs,
            max_passage_length=max_passage_length,
            weights_for_different_modes=weights_for_different_modes,
        )

    def load(self, file_path: str) -> "BGE":
        super().load(file_path)
        self._model = BGEM3FlagModel(self.model_path, use_fp16=self.use_fp16)
        return self
