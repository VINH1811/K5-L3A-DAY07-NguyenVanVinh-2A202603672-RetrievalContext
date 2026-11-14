"""Export Day 6 chunks (from Chroma) and benchmark questions to the Day 7 schema.

Run from the repo root:  python -X utf8 src/export_day6.py

Chunks: every record of the Day 6 collection -> id, text, source_id, source, page, metadata.
Questions: gold_page lists and `alternatives` are expanded into one gold option
per (source, page, evidence). Nothing is inferred; a gold option that is not
fully inside a chunk of the right source/page is reported as an error.
Unanswerable questions already present in data/questions.json are kept.
"""
import chromadb

from common import (CHROMA_PATH, CHUNKS_PATH, COLLECTION, DAY6_DIR, QUESTIONS_PATH,
                    hit, load_json, save_json)


def export_chunks() -> list[dict]:
    collection = chromadb.PersistentClient(path=str(CHROMA_PATH)).get_collection(COLLECTION)
    rows = collection.get(include=['documents', 'metadatas'])
    chunks = []
    for chunk_id, text, metadata in zip(rows['ids'], rows['documents'], rows['metadatas']):
        metadata = dict(metadata)
        chunks.append({'id': chunk_id, 'text': text, 'source_id': metadata.pop('document_id'),
                       'source': metadata.pop('source'), 'page': metadata.pop('page'),
                       'metadata': metadata})
    return sorted(chunks, key=lambda c: c['id'])


def gold_options(q: dict) -> list[dict]:
    options = []
    for item in [q, *q.get('alternatives', [])]:
        pages = item['gold_page'] if isinstance(item['gold_page'], list) else [item['gold_page']]
        options += [{'source_id': item['gold_source_id'], 'page': page,
                     'evidence': item['evidence_quote']} for page in pages]
    return options


def convert_question(q: dict) -> dict:
    out = {'id': q['id'], 'split': q['split'], 'type': q.get('type'), 'answerable': True,
           'question': q['question'], 'expected_answer': q['expected_answer'],
           'gold': gold_options(q)}
    if 'filter' in q:
        out['filter'] = q['filter']
    if 'note' in q:
        out['note'] = q['note']
    return out


def main() -> None:
    chunks = export_chunks()
    questions = [convert_question(q)
                 for q in load_json(DAY6_DIR / 'benchmark' / 'questions.json')['questions']]
    problems = [f"{q['id']}: gold option not inside any chunk: {g}"
                for q in questions for g in q['gold'] if not any(hit(c, [g]) for c in chunks)]
    if problems:
        raise SystemExit('\n'.join(problems))
    # Keep hand-written questions that do not come from Day 6 (e.g. unanswerable ones).
    day6_ids = {q['id'] for q in questions}
    extra = [q for q in load_json(QUESTIONS_PATH) if q['id'] not in day6_ids] if QUESTIONS_PATH.exists() else []
    save_json(CHUNKS_PATH, chunks)
    save_json(QUESTIONS_PATH, questions + extra)
    print(f'{len(chunks)} chunks from {COLLECTION} -> {CHUNKS_PATH}')
    print(f'{len(questions)} Day 6 questions + {len(extra)} extra -> {QUESTIONS_PATH}')


if __name__ == '__main__':
    main()
