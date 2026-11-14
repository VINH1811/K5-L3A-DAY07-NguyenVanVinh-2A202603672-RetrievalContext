"""Run retrieval evaluation or the full agent on the advanced benchmark.

python -X utf8 src/advanced/run_advanced.py --chunks mm --method hybrid --split dev --output starter/results/adv-dev-mm-hybrid
python -X utf8 src/advanced/run_advanced.py --chunks mm --method hybrid --split dev --retrieval-only --output ...

Agent: the same loop as the basic part (src/agent.py, context policy `interleave`) with the
Gemini controller below. With --page-images the decision step also receives the rendered page
images of the pages in the context (at most --max-images), so the model can read a chart or a
map itself instead of relying only on the extracted description.

Gold is read only after a question has finished, to mark which evidence groups the final context
contains. A context is sufficient when every evidence group has an item on a retrieved page; for
a table/figure item in the `mm` chunk set the chunk must be a table/figure chunk of that page.
"""
import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'src' / 'advanced'))
from common import load_json, save_json  # noqa: E402
from agent import run_agent  # noqa: E402
from controller import OllamaController  # noqa: E402
from gemini import default_model, generate, image_part  # noqa: E402

DATA = ROOT / 'starter' / 'data' / 'advanced'
QUESTIONS = ROOT / 'data' / 'advanced' / 'questions.json'


def page_of(chunk_id: str) -> int:
    return int(re.search(r':p(\d{3}):', chunk_id).group(1))


class GeminiController(OllamaController):
    """Same decision logic as the local controller; only the model call differs."""

    def __init__(self, model: str, *, max_searches: int, page_images: bool = False, max_images: int = 3):
        super().__init__(model, max_searches=max_searches)
        self.page_images, self.max_images = page_images, max_images
        self._pages: list[int] = []

    def decide(self, question, history):
        self._pages = []
        if history and self.page_images:
            pages = []
            for chunk_id in history[-1]['context_ids']:
                if page_of(chunk_id) not in pages:
                    pages.append(page_of(chunk_id))
            self._pages = pages[:self.max_images]
        return super().decide(question, history)

    def _chat(self, messages, **extra):
        system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
        parts = [{'text': m['content']} for m in messages if m['role'] == 'user']
        if self._pages and 'format' in extra:
            parts.append({'text': 'Page images of the pages the passages come from (page numbers are physical '
                                  f'PDF pages {self._pages}); use them to read tables, charts and maps. '
                                  'Cite only passage labels.'})
            parts += [image_part(DATA / 'pages' / f'p{p:03d}.png') for p in self._pages]
        tools = [t['function'] for t in extra.get('tools', [])]
        schema = extra.get('format')
        result = generate([{'role': 'user', 'parts': parts}], model=self.model, system=system,
                          tools=tools or None, schema=schema)
        usage = result['usage']
        self.calls.append({'model': result['model'], 'mode': 'tools' if tools else 'json', 'content': result['text'],
                           'tool_calls': result['function_calls'], 'page_images': list(self._pages) if schema else [],
                           'prompt_tokens': usage.get('promptTokenCount'),
                           'output_tokens': (usage.get('candidatesTokenCount') or 0) + (usage.get('thoughtsTokenCount') or 0),
                           'seconds': result['seconds'], 'retries': result['retries'], 'fallback': None})
        return {'content': result['text'],
                'tool_calls': [{'function': {'name': c.get('name'), 'arguments': c.get('args') or {}}}
                               for c in result['function_calls']]}


def build_retriever(chunk_set: str, method: str):
    from lexical import BM25Retriever
    chunks = load_json(DATA / f'chunks_{chunk_set}.json')
    if method == 'bm25':
        return BM25Retriever(chunks)
    from dense import DenseRetriever
    dense = DenseRetriever(collection=f'adv_{chunk_set}', path=DATA / 'chroma')
    if method == 'dense':
        return dense
    from hybrid import HybridRetriever
    return HybridRetriever([dense, BM25Retriever(chunks)])


def item_hit(chunk: dict, item: dict) -> bool:
    if chunk['page'] != item['page']:
        return False
    element = chunk['metadata'].get('element', 'text')
    if item['element'] == 'text' or chunk['metadata'].get('chunk_set') == 'pypdf':
        return True
    return element in ('table', 'figure')


def groups_covered(chunks: list[dict], question: dict) -> list[bool]:
    return [any(item_hit(c, item) for c in chunks for item in group) for group in question['evidence_groups']]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunks', choices=('pypdf', 'mm'), required=True)
    parser.add_argument('--method', choices=('bm25', 'dense', 'hybrid'), required=True)
    parser.add_argument('--split', choices=('dev', 'test'), required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--budget', type=int, default=9000, help='context budget (ký tự)')
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--page-images', action='store_true')
    parser.add_argument('--model', default=default_model())
    parser.add_argument('--retrieval-only', action='store_true')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    questions = [q for q in load_json(QUESTIONS) if q['split'] == args.split and (not args.ids or q['id'] in args.ids)]
    retriever = build_retriever(args.chunks, args.method)
    config = {'chunks': args.chunks, 'method': retriever.strategy, 'top_k': args.top_k, 'budget_chars': args.budget,
              'max_searches': args.max_searches, 'context_policy': 'interleave', 'page_images': args.page_images,
              'model': None if args.retrieval_only else args.model, 'split': args.split}

    if args.retrieval_only:
        traces = []
        for q in [q for q in questions if q['answerable']]:
            start = time.perf_counter()
            results = retriever.search(q['question'], filters=None, top_k=max(args.top_k, 3))
            ms = (time.perf_counter() - start) * 1000
            top3 = results[:3]
            first_group = lambda cs: groups_covered(cs, {'evidence_groups': q['evidence_groups'][:1]})[0]  # noqa: E731
            traces.append({'query_id': q['id'], 'group': q['group'], **config, 'elapsed_ms': ms,
                           'results': [{k: c[k] for k in ('id', 'page', 'score', 'rank')} | {'element': c['metadata'].get('element')} for c in results],
                           'hit1': first_group(top3[:1]), 'hit3': first_group(top3),
                           'context_all_groups': all(groups_covered(results[:args.top_k], q))})
        n = len(traces)
        summary = {**config, 'N': n, 'hit1': sum(t['hit1'] for t in traces), 'hit3': sum(t['hit3'] for t in traces),
                   'context_all_groups': sum(t['context_all_groups'] for t in traces),
                   'by_group': {g: f"{sum(t['context_all_groups'] for t in traces if t['group'] == g)}/{sum(t['group'] == g for t in traces)}"
                                for g in sorted({t['group'] for t in traces})},
                   'median_ms': statistics.median(t['elapsed_ms'] for t in traces)}
        save_json(args.output / 'traces.json', traces)
        save_json(args.output / 'summary.json', summary)
        print(json.dumps({k: summary[k] for k in ('N', 'hit1', 'hit3', 'context_all_groups', 'by_group', 'median_ms')}))
        return

    traces, sheet = [], []
    old = {t['query_id']: t for t in load_json(args.output / 'traces.json')} if args.ids and (args.output / 'traces.json').exists() else {}
    chunk_text = {c['id']: c for c in load_json(DATA / f'chunks_{args.chunks}.json')}
    for q in questions:
        controller = GeminiController(args.model, max_searches=args.max_searches, page_images=args.page_images)
        start = time.perf_counter()
        try:
            result = run_agent(q['question'], controller, retriever, max_searches=args.max_searches,
                               top_k=args.top_k, budget=args.budget, context_policy='interleave')
        except Exception as error:  # reported, never replaced by a fake answer
            result = {'status': 'error', 'answer': '', 'citations': [], 'events': [], 'error': f'{type(error).__name__}: {error}'}
        seconds = time.perf_counter() - start
        searches = [e for e in result['events'] if e['event'] == 'search']
        context_ids = searches[-1]['context_ids'] if searches else []
        trace = {'query_id': q['id'], 'split': q['split'], 'group': q['group'], 'answerable': q['answerable'],
                 'question': q['question'], **config, **result, 'n_searches': len(searches),
                 'queries': [e['query'] for e in searches], 'final_context_ids': context_ids,
                 'llm_calls': controller.calls, 'seconds': seconds,
                 'prompt_tokens': sum(c['prompt_tokens'] or 0 for c in controller.calls),
                 'output_tokens': sum(c['output_tokens'] or 0 for c in controller.calls)}
        if q['answerable']:
            covered = groups_covered([chunk_text[i] for i in context_ids], q)
            trace['context_groups'] = covered
            trace['context_sufficient'] = all(covered)
        traces.append(trace)
        sheet.append({'query_id': q['id'], 'group': q['group'], 'question': q['question'],
                      'expected': q.get('expected_answer') or q.get('expected_behavior'),
                      'evidence_groups': q.get('evidence_groups'), 'calculation': q.get('calculation'),
                      'status': result['status'], 'answer': result['answer'],
                      'citations': [{'id': i, 'page': chunk_text[i]['page'],
                                     'printed_page': chunk_text[i]['metadata']['printed_page'],
                                     'element': chunk_text[i]['metadata'].get('element'),
                                     'label': chunk_text[i]['metadata'].get('label'),
                                     'text': chunk_text[i]['text']} for i in result['citations']],
                      'pass': None, 'reason': None})
        print(f"{q['id']} {q['group']:12} {result['status']:8} s={len(searches)} {seconds:5.1f}s "
              f"ctx={trace.get('context_sufficient')} | {result['answer'][:120]!r}", flush=True)
    if old:
        order = [q['id'] for q in load_json(QUESTIONS)]
        new = {t['query_id'] for t in traces}
        traces = sorted([t for t in old.values() if t['query_id'] not in new] + traces, key=lambda t: order.index(t['query_id']))
        old_sheet = {r['query_id']: r for r in load_json(args.output / 'review_sheet.json')}
        sheet = sorted([r for k, r in old_sheet.items() if k not in new] + sheet, key=lambda r: order.index(r['query_id']))
    ans = [t for t in traces if t['answerable']]
    summary = {**config, 'N': len(traces), 'status': dict(Counter(t['status'] for t in traces)),
               'searches_distribution': dict(sorted(Counter(t['n_searches'] for t in traces).items())),
               'context_sufficient': f"{sum(t['context_sufficient'] for t in ans)}/{len(ans)}",
               'answerable_abstained': sum(t['status'] == 'abstain' for t in ans),
               'unanswerable_abstained': f"{sum(t['status'] == 'abstain' for t in traces if not t['answerable'])}/{len(traces) - len(ans)}",
               'errors': sum(t['status'] == 'error' for t in traces),
               'median_seconds': statistics.median(t['seconds'] for t in traces),
               'total_seconds': sum(t['seconds'] for t in traces),
               'model_calls': sum(len(t['llm_calls']) for t in traces),
               'prompt_tokens': sum(t['prompt_tokens'] for t in traces),
               'output_tokens': sum(t['output_tokens'] for t in traces)}
    save_json(args.output / 'traces.json', traces)
    save_json(args.output / 'review_sheet.json', sheet)
    save_json(args.output / 'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
