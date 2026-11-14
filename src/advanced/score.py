"""Aggregate human pass/fail labels of the advanced benchmark per group.

python -X utf8 src/advanced/score.py --run results/advanced/test-final

Reads `traces.json` and `reviews.json` (one {query_id, pass, reason} per question) from the run
directory and writes `score.json`. Every question of the split is in the denominator: a missing
label, an error or a timeout counts as a failure. Pass rate is not rounded.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

GROUPS = ('table', 'image', 'chart', 'multi-hop', 'text', 'unanswerable')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    traces = json.loads((args.run / 'traces.json').read_text(encoding='utf-8'))
    reviews = {r['query_id']: r for r in json.loads((args.run / 'reviews.json').read_text(encoding='utf-8'))}
    by_group = defaultdict(lambda: [0, 0])
    failures = []
    for t in traces:
        review = reviews.get(t['query_id'], {})
        passed = review.get('pass') is True and t['status'] != 'error'
        by_group[t['group']][0] += passed
        by_group[t['group']][1] += 1
        if not passed:
            failures.append({'query_id': t['query_id'], 'group': t['group'], 'status': t['status'],
                             'reason': review.get('reason', 'chưa có nhãn')})
    n = sum(v[0] for v in by_group.values())
    total = sum(v[1] for v in by_group.values())
    score = {'run': str(args.run), 'passed': n, 'total': total, 'rate_percent': 100 * n / total,
             'meets_90_percent': n * 10 >= total * 9,
             'by_group': {g: f'{by_group[g][0]}/{by_group[g][1]}' for g in GROUPS if g in by_group},
             'errors': sum(t['status'] == 'error' for t in traces), 'failures': failures}
    (args.run / 'score.json').write_text(json.dumps(score, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in score.items() if k != 'failures'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
