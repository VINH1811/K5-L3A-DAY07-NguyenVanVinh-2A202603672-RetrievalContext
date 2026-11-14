"""Build the two chunk sets of the advanced challenge and index them in Chroma.

python -X utf8 src/advanced/build_index.py

- `pypdf`: text layer only (pypdf), split per page into chunks of at most 900 characters.
  This is the Day 6 style baseline: charts become loose numbers, maps and diagrams are lost.
- `mm` (multimodal): elements extracted by the vision model (src/advanced/extract.py). Text is
  split into chunks of at most 900 characters; every table and every figure is ONE chunk whose
  text starts with its label and title, so it can be cited as a whole.

Every chunk keeps the physical page, the printed page, the element type and label, so a citation
points to a page region that can be checked on the original. Embeddings: google/embeddinggemma-2
(same model and prefixes as Day 6), stored in Chroma (cosine) under starter/data/advanced/chroma.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from common import save_json  # noqa: E402

DATA = ROOT / 'starter' / 'data' / 'advanced'
SOURCE_ID = 'wb-vn-ccdr-2022'
SOURCE = ('https://documents1.worldbank.org/curated/en/099355507062229876/pdf/'
          'P17724103036ee07909701028a237b6d18b.pdf')
MAX_CHARS = 900


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split on sentence ends, keeping chunks under `limit` characters."""
    sentences = re.split(r'(?<=[.!?;:])\s+', re.sub(r'\s+', ' ', text).strip())
    chunks, current = [], ''
    for sentence in sentences:
        while len(sentence) > limit:  # very long run-on text (e.g. a table read as text)
            if current:
                chunks.append(current)
                current = ''
            chunks.append(sentence[:limit])
            sentence = sentence[limit:]
        if current and len(current) + 1 + len(sentence) > limit:
            chunks.append(current)
            current = sentence
        else:
            current = f'{current} {sentence}'.strip()
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 40]


def chunk(kind: str, page: int, n: int, text: str, element: str, label: str = '', title: str = '',
          printed: str = '') -> dict:
    return {'id': f'ccdr:p{page:03d}:{kind}:{n:02d}', 'text': text, 'source_id': SOURCE_ID, 'source': SOURCE,
            'page': page, 'metadata': {'printed_page': printed or str(page - 10), 'element': element,
                                       'label': label, 'title': title, 'chunk_set': kind.split('-')[0]}}


def pypdf_chunks() -> list[dict]:
    out = []
    for row in json.loads((DATA / 'pypdf_pages.json').read_text(encoding='utf-8')):
        for n, text in enumerate(split_text(row['text'])):
            out.append(chunk('pypdf', row['page'], n, text, 'text'))
    return out


def mm_chunks() -> list[dict]:
    out = []
    for path in sorted((DATA / 'extracted').glob('p*.json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        page, printed, n = row['page'], row.get('page_number_printed', ''), 0
        for element in row['elements']:
            content = element['content'].strip()
            if not content:
                continue
            if element['type'] == 'text':
                for piece in split_text(content):
                    out.append(chunk('mm-text', page, n, piece, 'text', printed=printed))
                    n += 1
            else:
                head = ' '.join(x for x in (element['label'], element['title']) if x).strip()
                text = f"[{element['type'].upper()}: {head or 'untitled'} | page {printed or page - 10}]\n{content}"
                out.append(chunk(f"mm-{element['type']}", page, n, text, element['type'],
                                 element['label'], element['title'], printed))
                n += 1
    return out


def embed_and_store(chunks: list[dict], collection_name: str) -> None:
    import chromadb
    import torch
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer('google/embeddinggemma-2', config_kwargs={'vision_config': None, 'audio_config': None},
                                model_kwargs={'dtype': torch.float32})
    vectors = model.encode([f"title: none | text: {c['text']}" for c in chunks], batch_size=16,
                           normalize_embeddings=True, show_progress_bar=False)
    client = chromadb.PersistentClient(path=str(DATA / 'chroma'))
    try:
        client.delete_collection(collection_name)
    except Exception:  # collection did not exist yet
        pass
    collection = client.create_collection(collection_name, metadata={'hnsw:space': 'cosine', 'dimensions': int(vectors.shape[1])})
    collection.add(ids=[c['id'] for c in chunks], embeddings=[[float(x) for x in v] for v in vectors],
                   documents=[c['text'] for c in chunks],
                   metadatas=[{**c['metadata'], 'document_id': c['source_id'], 'source': c['source'], 'page': c['page']}
                              for c in chunks])


def main() -> None:
    import time
    for name, build in (('pypdf', pypdf_chunks), ('mm', mm_chunks)):
        chunks = build()
        if not chunks:
            print(f'{name}: no chunks (run extract.py first)')
            continue
        assert len({c['id'] for c in chunks}) == len(chunks)
        save_json(DATA / f'chunks_{name}.json', chunks)
        started = time.perf_counter()
        embed_and_store(chunks, f'adv_{name}')
        kinds = {}
        for c in chunks:
            kinds[c['metadata']['element']] = kinds.get(c['metadata']['element'], 0) + 1
        print(f"{name}: {len(chunks)} chunks {kinds}, pages {len({c['page'] for c in chunks})}, "
              f"embed+store {time.perf_counter() - started:.0f}s", flush=True)


if __name__ == '__main__':
    main()
