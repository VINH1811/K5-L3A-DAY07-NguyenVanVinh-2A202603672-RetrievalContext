"""Shared paths, JSON helpers and the evidence-hit definition for Day 7."""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'starter'))  # reuse starter/baseline.py helpers

DAY6_DIR = Path(os.environ.get(
    'DAY6_DIR', ROOT.parent / 'K5-L3A-DAY06-NguyenVanVinh-2A202603672-DataIngestion'))
CHROMA_PATH = Path(os.environ.get('DAY6_CHROMA_PATH', DAY6_DIR / 'starter' / 'chroma_db'))
COLLECTION = os.environ.get('DAY6_COLLECTION', 'day6_B')  # by_paragraph, max 800 chars
EMBEDDING_MODEL = os.environ.get('EMBEDDING_MODEL', 'google/embeddinggemma-2')

# Chunk text comes from copyrighted World Bank PDFs: kept in a git-ignored folder
# and re-created with src/export_day6.py. Questions are our own and are committed.
CHUNKS_PATH = ROOT / 'starter' / 'data' / 'day7' / 'chunks.json'
QUESTIONS_PATH = ROOT / 'data' / 'questions.json'


def load_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path: Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def match_text(text: str) -> str:
    """Same normalization as Day 6: NFC, collapse whitespace, casefold.

    PDF text keeps line breaks inside sentences, so a raw substring test would
    miss evidence that is really there.
    """
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip().casefold()


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """A chunk is a hit when it has the source and page of one gold option and
    contains that option's evidence quote (after match_text). Gold options are
    alternatives: any one is enough."""
    text = match_text(chunk['text'])
    return any(chunk['source_id'] == g['source_id'] and chunk['page'] == g['page']
               and match_text(g['evidence']) in text for g in gold)
