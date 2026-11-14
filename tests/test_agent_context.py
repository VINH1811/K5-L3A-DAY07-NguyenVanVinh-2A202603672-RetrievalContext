"""Context policy of src/agent.py. Controller and retriever are mocks: code tests only."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from agent import order_candidates, run_agent  # noqa: E402


def chunk(i):
    return {'id': i, 'text': 'x' * 60, 'source_id': 's', 'source': 'src', 'page': 1, 'metadata': {}}


class Retriever:
    def __init__(self, pages):
        self.pages = iter(pages)

    def search(self, query, *, filters=None, top_k=3):
        return [chunk(i) for i in next(self.pages)][:top_k]


class Controller:
    """Searches twice, then cites the last chunk it was shown."""
    def decide(self, question, history):
        queries = history[-1]['queries'] if history else []
        if len(queries) < 2:
            return {'action': 'search', 'query': f'q{len(queries)}'}
        ids = history[-1]['context_ids']
        return {'action': 'answer', 'answer': f'ok [{ids[0]}]', 'citations': [ids[0]]}


class ContextPolicyTests(unittest.TestCase):
    def test_order(self):
        searches = [[chunk('a1'), chunk('a2')], [chunk('b1'), chunk('b2')]]
        self.assertEqual([c['id'] for c in order_candidates(searches, 'first_come')], ['a1', 'a2', 'b1', 'b2'])
        self.assertEqual([c['id'] for c in order_candidates(searches, 'interleave')], ['b1', 'a1', 'b2', 'a2'])

    def run_policy(self, policy):
        # budget fits two blocks of ~80 chars
        return run_agent('q?', Controller(), Retriever([['a1', 'a2'], ['b1', 'b2']]),
                         top_k=2, budget=170, context_policy=policy)

    def test_first_come_keeps_first_search_only(self):
        result = self.run_policy('first_come')
        self.assertEqual(result['events'][1]['context_ids'], ['a1', 'a2'])

    def test_interleave_lets_new_search_in(self):
        result = self.run_policy('interleave')
        self.assertEqual(result['events'][1]['context_ids'], ['b1', 'a1'])
        self.assertEqual(result['status'], 'answered')
        self.assertEqual(result['citations'], ['b1'])

    def test_duplicate_chunk_counted_once(self):
        result = run_agent('q?', Controller(), Retriever([['a1', 'a2'], ['a1', 'b2']]),
                           top_k=2, budget=170, context_policy='interleave')
        self.assertEqual(result['events'][1]['context_ids'], ['a1', 'b2'])

    def test_bad_policy(self):
        with self.assertRaises(ValueError):
            self.run_policy('newest')


if __name__ == '__main__':
    unittest.main()
