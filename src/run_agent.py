"""Run the agent (real model + real retriever) on one split and save traces.

Run from the repo root, e.g.:
  python -X utf8 src/run_agent.py --method hybrid --split dev --output starter/results/agent-dev-hybrid

Writes to --output:
  traces.json        one record per question: status, answer, citations, events, model calls, time
  cases.json         id / split / answerable, for starter/agent_metrics.py
  review_sheet.json  what a human needs to grade each answer; labels are left empty (null)
  summary.json       counts that need no human judgement (status, tool calls, context evidence, time)

Gold is read only after the agent has finished a question, to mark whether the
final context contained valid evidence. It never reaches the model or retriever.
"""
import argparse
import statistics
import time
from collections import Counter
from pathlib import Path

from common import QUESTIONS_PATH, hit, load_json, save_json  # adds starter/ to sys.path
from agent import POLICIES, run_agent
from controller import OllamaController
from eval_retrieval import build


def main() -> None:
    parser = argparse.ArgumentParser(description='Chạy agent thật trên một split')
    parser.add_argument('--method', required=True, choices=('tfidf', 'bm25', 'dense', 'hybrid'))
    parser.add_argument('--split', required=True, choices=('dev', 'test'))
    parser.add_argument('--model', default='qwen2.5:7b')
    parser.add_argument('--top-k', type=int, default=3)
    parser.add_argument('--budget', type=int, default=3000, help='context budget, tính bằng ký tự')
    parser.add_argument('--max-searches', type=int, default=3)
    parser.add_argument('--num-gpu', type=int, default=0,
                        help='số layer chạy GPU của Ollama; 0 = chỉ CPU (mặc định, vì GPU runner lỗi CUDA trên máy này), -1 = để Ollama tự chọn')
    parser.add_argument('--context-policy', choices=POLICIES, default='first_come',
                        help='cách chọn context qua nhiều lượt tìm (xem src/agent.py)')
    parser.add_argument('--use-filter', action='store_true',
                        help='dùng filter do người dùng cung cấp trong câu hỏi (nếu có)')
    parser.add_argument('--ids', nargs='*',
                        help='chỉ chạy các câu này; nếu --output đã có kết quả cùng cấu hình thì gộp vào')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()

    questions = [q for q in load_json(QUESTIONS_PATH) if q['split'] == args.split]
    if args.ids:
        questions = [q for q in questions if q['id'] in args.ids]
    retriever = build(args.method)
    config = {'model': args.model, 'method': retriever.strategy, 'top_k': args.top_k,
              'budget_chars': args.budget, 'max_searches': args.max_searches, 'context_policy': args.context_policy,
              'use_filter': args.use_filter, 'split': args.split, 'num_gpu': args.num_gpu}

    traces, sheet = [], []
    for q in questions:
        answerable = q.get('answerable', True)
        filters = q.get('filter') if args.use_filter else None
        controller = OllamaController(args.model, max_searches=args.max_searches,
                                      num_gpu=None if args.num_gpu < 0 else args.num_gpu)
        start = time.perf_counter()
        try:
            result = run_agent(q['question'], controller, retriever, filters=filters,
                               max_searches=args.max_searches, top_k=args.top_k, budget=args.budget,
                               context_policy=args.context_policy)
        except Exception as error:  # reported as an error, never replaced by a fake result
            result = {'status': 'error', 'answer': '', 'citations': [], 'events': [],
                      'error': f'{type(error).__name__}: {error}'}
        seconds = time.perf_counter() - start
        searches = [e for e in result['events'] if e['event'] == 'search']
        context_ids = searches[-1]['context_ids'] if searches else []
        chunks = {c['id']: c for e in searches for c in e['results']}
        trace = {'query_id': q['id'], 'split': q['split'], 'answerable': answerable,
                 'question': q['question'], **config, 'filter': filters, **result,
                 'n_searches': len(searches), 'queries': [e['query'] for e in searches],
                 'final_context_ids': context_ids, 'llm_calls': controller.calls,
                 'prompt_tokens': sum(c['prompt_tokens'] or 0 for c in controller.calls),
                 'output_tokens': sum(c['output_tokens'] or 0 for c in controller.calls),
                 'seconds': seconds, 'service_cost_usd': 0}
        if answerable:  # scoring only, after the run
            trace['context_hit'] = any(hit(chunks[i], q['gold']) for i in context_ids)
        traces.append(trace)
        sheet.append({'query_id': q['id'], 'answerable': answerable, 'question': q['question'],
                      'expected_answer': q.get('expected_answer'), 'gold': q.get('gold'),
                      'absence_reason': q.get('absence_reason'),
                      'status': result['status'], 'answer': result['answer'],
                      'citations': [{'id': i, 'page': chunks[i]['page'], 'source_id': chunks[i]['source_id'],
                                     'text': chunks[i]['text']} for i in result['citations']],
                      # to be filled in by a human reader:
                      'answer_grade': None, 'citation_support': [None] * len(result['citations']),
                      'abstention_correct': None, 'reason': None})
        print(f"{q['id']} {result['status']:8} searches={len(searches)} {seconds:5.1f}s "
              f"ctx_hit={trace.get('context_hit')} | {result['answer'][:110]!r}")

    if args.ids and (args.output / 'traces.json').exists():
        # Add or replace only the requested questions in an existing run of the same configuration.
        old_traces = load_json(args.output / 'traces.json')
        if any({k: t.get(k) for k in config} != config for t in old_traces):
            raise SystemExit('--output chứa kết quả của cấu hình khác; không gộp')
        new_ids = {t['query_id'] for t in traces}
        order = [q['id'] for q in load_json(QUESTIONS_PATH)]
        traces = sorted([t for t in old_traces if t['query_id'] not in new_ids] + traces,
                        key=lambda t: order.index(t['query_id']))
        sheet = sorted([r for r in load_json(args.output / 'review_sheet.json') if r['query_id'] not in new_ids] + sheet,
                       key=lambda r: order.index(r['query_id']))
    answerable_traces = [t for t in traces if t['answerable']]
    summary = {**config, 'N': len(traces), 'status': dict(Counter(t['status'] for t in traces)),
               'tool_use': sum(t['n_searches'] > 0 for t in traces),
               'searches_distribution': dict(sorted(Counter(t['n_searches'] for t in traces).items())),
               'fallback_first_search': sum(any(c['fallback'] for c in t['llm_calls']) for t in traces),
               'context_hit': sum(t['context_hit'] for t in answerable_traces),
               'answerable_N': len(answerable_traces),
               'answerable_abstained': sum(t['status'] == 'abstain' for t in answerable_traces),
               'unanswerable_abstained': sum(t['status'] == 'abstain' for t in traces if not t['answerable']),
               'unanswerable_N': len(traces) - len(answerable_traces),
               'median_seconds': statistics.median(t['seconds'] for t in traces),
               'total_seconds': sum(t['seconds'] for t in traces),
               'prompt_tokens': sum(t['prompt_tokens'] for t in traces),
               'output_tokens': sum(t['output_tokens'] for t in traces), 'service_cost_usd': 0}
    save_json(args.output / 'traces.json', traces)
    save_json(args.output / 'cases.json', [{'id': t['query_id'], 'split': t['split'],
                                            'answerable': t['answerable']} for t in traces])
    save_json(args.output / 'review_sheet.json', sheet)
    save_json(args.output / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    main()
