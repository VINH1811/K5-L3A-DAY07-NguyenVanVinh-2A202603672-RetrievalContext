"""Agent loop for Day 7: starter/agentic.py's run_agent plus a context selection policy.

Why this exists: the starter packs candidates in arrival order, so chunks from the
first search fill the budget and later searches cannot add anything (seen on dev
q09: three different queries, same three chunks in the context every time).

context_policy:
- 'first_come'  identical to the starter: earlier searches first.
- 'interleave'  order by rank inside each search, newest search first on ties
                (s3 r1, s2 r1, s1 r1, s3 r2, ...), then deduplicate and pack. The
                best result of every search gets a place; a new search is preferred
                because the model asked for it after judging the old passages insufficient.

Everything else is unchanged: no gold enters the loop, a final answer without valid
citations or an exhausted search limit leads to abstention, errors propagate.
"""
from typing import Any

from common import ROOT  # noqa: F401  (adds starter/ to sys.path)
from baseline import pack_context  # noqa: E402

POLICIES = ('first_come', 'interleave')
ABSTAIN_TEXT = 'Không đủ bằng chứng để trả lời.'


def order_candidates(searches: list[list[dict[str, Any]]], policy: str) -> list[dict[str, Any]]:
    """`searches` holds the result list of each search, oldest first."""
    if policy == 'first_come':
        return [c for results in searches for c in results]
    keyed = [(position, -index, c) for index, results in enumerate(searches)
             for position, c in enumerate(results)]
    return [c for _, _, c in sorted(keyed, key=lambda item: item[:2])]


def run_agent(question: str, controller, retriever, *, filters: dict[str, Any] | None = None,
              max_searches: int = 3, top_k: int = 3, budget: int = 3000,
              context_policy: str = 'first_come') -> dict[str, Any]:
    if not question.strip() or max_searches < 1 or top_k < 1 or budget < 1:
        raise ValueError('question, max_searches, top_k hoặc budget không hợp lệ')
    if context_policy not in POLICIES:
        raise ValueError(f'context_policy phải thuộc {POLICIES}')
    history: list[dict[str, Any]] = []
    chunks: dict[str, dict[str, Any]] = {}
    searches: list[list[dict[str, Any]]] = []
    events: list[dict[str, Any]] = []

    def abstain(event: str | None = None) -> dict[str, Any]:
        return {'status': 'abstain', 'answer': ABSTAIN_TEXT, 'citations': [],
                'events': events + ([{'event': event}] if event else [])}

    for _ in range(max_searches + 1):
        action = controller.decide(question, history)
        if action.get('action') == 'search':
            if len(searches) >= max_searches:
                return abstain('search_limit')
            query = action.get('query')
            if not isinstance(query, str) or not query.strip():
                raise ValueError('search query phải là chuỗi không rỗng')
            results = retriever.search(query, filters=filters, top_k=top_k)
            if len(results) > top_k or len({r['id'] for r in results}) != len(results):
                raise ValueError('retriever phải trả tối đa top_k chunk ID duy nhất')
            searches.append(results)
            candidates = order_candidates(searches, context_policy)
            context, ids = pack_context(candidates, budget=budget)
            chunks = {c['id']: c for c in candidates if c['id'] in ids}
            events.append({'event': 'search', 'query': query, 'filter': filters, 'results': results,
                           'context': context, 'context_ids': ids, 'context_policy': context_policy})
            # Send only the latest cumulative context; avoid multiplying the budget.
            history = [{'tool': 'search', 'queries': [e['query'] for e in events],
                        'context': context, 'context_ids': ids}]
        elif action.get('action') == 'answer':
            answer, citations = action.get('answer'), action.get('citations')
            if not isinstance(answer, str) or not isinstance(citations, list) or any(not isinstance(i, str) for i in citations):
                raise ValueError('answer/citations không hợp lệ')
            if answer.strip() and citations and all(i in chunks and f'[{i}]' in answer for i in citations):
                return {'status': 'answered', 'answer': answer, 'citations': citations,
                        'events': events + [{'event': 'answer'}]}
            return abstain('invalid_or_missing_citations')
        else:
            raise ValueError('action phải là search hoặc answer')
    return abstain()
