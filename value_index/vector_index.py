# value_index/vector_index.py
from dataclasses import dataclass

import faiss
from sentence_transformers import SentenceTransformer


@dataclass
class ColumnVectorIndex:
    index: faiss.IndexFlatL2
    original_strings: list[str]

    def get_similar_strings(self, emb_model: SentenceTransformer, query_string: str, k: int = 3) -> list[str]:
        query_embedding = emb_model.encode([query_string])
        results = self.index.search(query_embedding, k=min(k, len(self.original_strings)))
        return [self.original_strings[i] for i in results[1][0]]
