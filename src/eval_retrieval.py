"""Evaluate one retrieval method on one split: Hit@1, Evidence Hit@3, Context Evidence Rate.

Run from the repo root, e.g.:
  python -X utf8 src/eval_retrieval.py --method dense --split dev --output starter/results/dev-dense

Only answerable questions are scored (unanswerable ones are not in the denominator).
Gold is used only for scoring after retrieval; it never reaches the retriever.
"""
import argparse
import statistics
import time
from pathlib import Path

from common import CHUNKS_PATH, QUESTIONS_PATH, hit, load_json, save_json  # adds starter/ to sys.path
from baseline import TfidfIndex, pack_context  # noqa: E402

RANK_CUTOFF = 3  # every method reports top-3 so rankings are comparable


class TfidfRetriever:
    strategy = 'tfidf'

    def __init__(self) -> None:
        self._index = TfidfIndex(load_json(CHUNKS_PATH))

    def search(self, query, *, filters=None, top_k=3):
        if filters:
            raise ValueError('tfidf baseline không hỗ trợ filter')
        results = self._index.search(query, top_k=top_k)
        return [{**c, 'rank': rank} for rank, c in enumerate(results, start=1)]


def build(method: str):
    if method == 'tfidf':
        return TfidfRetriever()
    if method == 'bm25':
        from lexical import BM25Retriever
        return BM25Retriever()
    if method == 'dense':
        from dense import DenseRetriever
        return DenseRetriever()
    if method == 'hybrid':
        from dense import DenseRetriever
        from hybrid import HybridRetriever
        from lexical import BM25Retriever
        return HybridRetriever([DenseRetriever(), BM25Retriever()])
    raise ValueError(f'chưa có phương pháp {method}')


def evaluate(retriever, questions, *, top_k: int, budget: int, use_filter: bool):
    traces = []
    for q in questions:
        filters = q.get('filter') if use_filter else None
        start = time.perf_counter()
        results = retriever.search(q['question'], filters=filters, top_k=max(top_k, RANK_CUTOFF))
        context, ids = pack_context(results[:top_k], budget)
        elapsed_ms = (time.perf_counter() - start) * 1000
        marked = [{**c, 'hit': hit(c, q['gold'])} for c in results]
        top3 = marked[:RANK_CUTOFF]
        traces.append({'query_id': q['id'], 'split': q['split'], 'question': q['question'],
                       'strategy': retriever.strategy, 'top_k': top_k, 'filter': filters,
                       'budget_chars': budget, 'gold': q['gold'], 'results': marked,
                       'context': context, 'context_ids': ids, 'context_chars': len(context),
                       'elapsed_ms': elapsed_ms,
                       'hit1': bool(top3 and top3[0]['hit']),
                       'hit3': any(c['hit'] for c in top3),
                       'context_hit': any(c['hit'] and c['id'] in ids for c in marked)})
    return traces


def main() -> None:
    parser = argparse.ArgumentParser(description='Đánh giá một phương pháp retrieval trên một split')
    parser.add_argument('--method', required=True, choices=('tfidf', 'bm25', 'dense', 'hybrid'))
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--top-k', type=int, default=3, help='số chunk đưa vào context')
    parser.add_argument('--budget', type=int, default=800, help='context budget, tính bằng ký tự')
    parser.add_argument('--use-filter', action='store_true',
                        help='dùng filter do người dùng cung cấp trong câu hỏi (nếu có)')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    questions = [q for q in load_json(QUESTIONS_PATH)
                 if q['split'] == args.split and q.get('answerable', True)]
    started = time.perf_counter()
    retriever = build(args.method)
    load_seconds = time.perf_counter() - started
    traces = evaluate(retriever, questions, top_k=args.top_k, budget=args.budget,
                      use_filter=args.use_filter)
    n = len(traces)
    summary = {'strategy': retriever.strategy, 'split': args.split, 'N': n,
               'top_k': args.top_k, 'budget_chars': args.budget, 'use_filter': args.use_filter,
               'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
               'context_hit': sum(t['context_hit'] for t in traces),
               'avg_context_chars': statistics.mean(t['context_chars'] for t in traces),
               'median_ms': statistics.median(t['elapsed_ms'] for t in traces),
               'load_seconds': load_seconds, 'service_cost_usd': 0}
    save_json(args.output / 'traces.json', traces)
    save_json(args.output / 'summary.json', summary)
    print(f"{retriever.strategy} {args.split}: Hit@1 {summary['hit1']}/{n} | "
          f"Hit@3 {summary['hit3']}/{n} | Context {summary['context_hit']}/{n} | "
          f"median {summary['median_ms']:.1f} ms")
    for t in traces:
        if not t['context_hit']:
            print(f"  miss {t['query_id']}: hit3={t['hit3']} context_ids={t['context_ids']}")


if __name__ == '__main__':
    main()
