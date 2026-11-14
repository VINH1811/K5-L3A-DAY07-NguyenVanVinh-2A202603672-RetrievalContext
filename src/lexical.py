"""Lexical retrieval: Okapi BM25 over the exported Day 6 chunks. Standard library only.

Tokenization is the starter's (NFC, casefold, \\w+): no stemming, no stop words.
"""
import math
from collections import Counter
from typing import Any

from common import CHUNKS_PATH, load_json  # adds starter/ to sys.path
from baseline import tokenize  # noqa: E402


class BM25Retriever:
    strategy = 'bm25'

    def __init__(self, chunks: list[dict[str, Any]] | None = None, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.chunks = chunks if chunks is not None else load_json(CHUNKS_PATH)
        self.k1, self.b = k1, b
        self._counts = [Counter(tokenize(c['text'])) for c in self.chunks]
        self._lengths = [sum(row.values()) for row in self._counts]
        self._avg_length = sum(self._lengths) / len(self._lengths)
        df = Counter(term for row in self._counts for term in row)
        n = len(self.chunks)
        self._idf = {term: math.log(1 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}

    def search(self, query: str, *, filters: dict[str, Any] | None = None,
               top_k: int = 3) -> list[dict[str, Any]]:
        """Return up to top_k chunks with a positive BM25 score.

        `filters` is the same metadata equality filter as the dense retriever.
        """
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        scored = []
        for chunk, counts, length in zip(self.chunks, self._counts, self._lengths):
            if filters and any(chunk['metadata'].get(k) != v for k, v in filters.items()):
                continue
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_length)
            score = sum(self._idf[t] * counts[t] * (self.k1 + 1) / (counts[t] + norm)
                        for t in terms if t in counts)
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda pair: (-pair[0], pair[1]['id']))
        return [{**chunk, 'score': score, 'rank': rank, 'strategy': self.strategy}
                for rank, (score, chunk) in enumerate(scored[:top_k], start=1)]
