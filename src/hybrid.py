"""Hybrid retrieval: dense + BM25 merged with Reciprocal Rank Fusion (RRF).

RRF uses ranks only, so cosine similarity and BM25 scores (different scales)
are never added together: score(chunk) = sum over retrievers of 1 / (k + rank).
"""
from typing import Any


class HybridRetriever:
    strategy = 'hybrid_rrf'

    def __init__(self, retrievers: list[Any], *, candidates: int = 20, rrf_k: int = 60) -> None:
        if len(retrievers) < 2 or candidates < 1 or rrf_k < 0:
            raise ValueError('cần ít nhất 2 retriever, candidates >= 1 và rrf_k >= 0')
        self.retrievers, self.candidates, self.rrf_k = retrievers, candidates, rrf_k

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        """Take `candidates` results from each retriever, fuse by RRF, return top_k.

        Each result keeps the rank and score it had in every retriever under `components`.
        """
        fused: dict[str, dict[str, Any]] = {}
        for retriever in self.retrievers:
            for c in retriever.search(query, filters=filters, top_k=max(self.candidates, top_k)):
                entry = fused.setdefault(c['id'], {
                    **{k: c[k] for k in ('id', 'text', 'source_id', 'source', 'page', 'metadata')},
                    'score': 0.0, 'components': {}})
                entry['score'] += 1 / (self.rrf_k + c['rank'])
                entry['components'][retriever.strategy] = {'rank': c['rank'], 'score': c['score']}
        ranked = sorted(fused.values(), key=lambda c: (-c['score'], c['id']))[:top_k]
        return [{**c, 'rank': rank, 'strategy': self.strategy} for rank, c in enumerate(ranked, start=1)]
