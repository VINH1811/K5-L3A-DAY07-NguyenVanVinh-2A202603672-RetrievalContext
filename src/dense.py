"""Dense retrieval on the Day 6 Chroma vector store (cosine, HNSW).

The query is embedded with the same model and query prefix used to build the
index in Day 6 (google/embeddinggemma-2, 768 dimensions, L2-normalized).
Errors from the model or the store propagate; there is no mock fallback.
"""
from typing import Any

import chromadb

from common import CHROMA_PATH, COLLECTION, EMBEDDING_MODEL

QUERY_PREFIX = 'task: search result | query: '


class DenseRetriever:
    strategy = 'dense'

    def __init__(self, model_name: str = EMBEDDING_MODEL, collection: str = COLLECTION,
                 path=CHROMA_PATH) -> None:
        import torch
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        if 'embeddinggemma-2' not in model_name.lower():
            raise ValueError('index Day 6 được tạo bằng embeddinggemma-2; đổi model cần index mới')
        self.model_name = model_name
        # Text-only load in float32, exactly as in Day 6 src/embeddings.py.
        self._model = SentenceTransformer(
            model_name, config_kwargs={'vision_config': None, 'audio_config': None},
            model_kwargs={'dtype': torch.float32})
        self._collection = chromadb.PersistentClient(path=str(path)).get_collection(collection)
        self.dimensions = int(self._collection.metadata['dimensions'])

    def embed_query(self, query: str) -> list[float]:
        vector = self._model.encode([QUERY_PREFIX + query], normalize_embeddings=True,
                                    show_progress_bar=False)[0]
        if len(vector) != self.dimensions:
            raise ValueError(f'query có {len(vector)} chiều, index cần {self.dimensions}')
        return [float(x) for x in vector]

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        """Return up to top_k chunks ranked by cosine similarity (score = 1 - distance).

        `filters` is a metadata equality filter (e.g. {'year': 2023}) applied inside Chroma.
        """
        conditions = [{key: value} for key, value in (filters or {}).items()]
        where = None if not conditions else conditions[0] if len(conditions) == 1 else {'$and': conditions}
        result = self._collection.query(
            query_embeddings=[self.embed_query(query)], n_results=top_k, where=where,
            include=['documents', 'metadatas', 'distances'])
        hits = []
        for rank, (chunk_id, text, metadata, distance) in enumerate(
                zip(result['ids'][0], result['documents'][0], result['metadatas'][0],
                    result['distances'][0]), start=1):
            metadata = dict(metadata)
            hits.append({'id': chunk_id, 'text': text, 'source_id': metadata.pop('document_id'),
                         'source': metadata.pop('source'), 'page': metadata.pop('page'),
                         'metadata': metadata, 'score': 1 - float(distance), 'rank': rank,
                         'strategy': self.strategy})
        return hits
