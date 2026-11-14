"""Extract every page of the PDF into typed elements (text / table / figure) with a vision model.

python -X utf8 src/advanced/extract.py --pages starter/data/advanced/pages --pypdf starter/data/advanced/pypdf_pages.json --out starter/data/advanced/extracted

For each page the model receives the page image and the raw pypdf text layer, and returns JSON:
body text (cleaned), each table transcribed with headers and units, and each figure/map/diagram
described with its title, axes, legend, units and every readable value or label. One JSON file
per page is cached, so the run can be resumed. Extracted content is model output: the benchmark
evidence was checked against the page images, not against this file.
"""
import argparse
import json
from pathlib import Path

from gemini import default_model, generate, image_part

PROMPT = """You are digitizing page {page} of a World Bank report so that it can be searched.
Use the page image as the source of truth; the raw text layer below is only a help (it can be
out of order or miss text inside charts).

Return JSON with:
- "page_number_printed": the page number printed on the page, or "" if none.
- "elements": the content of the page in reading order. Each element has
  - "type": "text", "table" or "figure" (figure = chart, map, diagram, infographic).
  - "label": e.g. "Table 2", "Figure 13", "Box 7", or "" for plain text.
  - "title": the caption/title, or "".
  - "content":
    * text: the body text and footnotes of that part of the page, faithfully, without page headers.
    * table: every row as "row name | column: value | column: value ...", keeping units, ranges and signs exactly.
    * figure: what it shows; axis titles and units; legend. Then "Printed values:" listing EVERY number
      or label printed on the figure together with exactly what it labels (series, category, year), e.g.
      "Coal segment of the CPS bar: 276 (TWh)". Only if useful, then "Approximate readings (~):" for
      unlabeled values read off the axes, each marked with ~. Never present an estimate as a printed value
      and never fill in segments that carry no label. For maps: the legend classes and which regions/places
      are highlighted and how strongly, plus every place name printed. For diagrams: all boxes, arrows and
      their text. Describe visible trends (rising, falling, highest, lowest).
Ignore purely decorative photos (say so in one short figure element if a photo fills the page).

Raw text layer:
{text}"""

SCHEMA = {'type': 'object', 'required': ['page_number_printed', 'elements'], 'properties': {
    'page_number_printed': {'type': 'string'},
    'elements': {'type': 'array', 'items': {'type': 'object', 'required': ['type', 'label', 'title', 'content'],
                 'properties': {'type': {'type': 'string', 'enum': ['text', 'table', 'figure']},
                                'label': {'type': 'string'}, 'title': {'type': 'string'},
                                'content': {'type': 'string'}}}}}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pages', type=Path, required=True)
    parser.add_argument('--pypdf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default=default_model())
    parser.add_argument('--only', type=int, nargs='*', help='chỉ xử lý các trang này')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    texts = {p['page']: p['text'] for p in json.loads(args.pypdf.read_text(encoding='utf-8'))}
    for page in sorted(args.only or texts):
        target = args.out / f'p{page:03d}.json'
        if target.exists():
            continue
        result = generate([{'role': 'user', 'parts': [
            {'text': PROMPT.format(page=page, text=texts[page][:12000] or '(empty)')},
            image_part(args.pages / f'p{page:03d}.png')]}], model=args.model, schema=SCHEMA)
        data = json.loads(result['text'])
        record = {'page': page, **data, 'model': result['model'], 'usage': result['usage'],
                  'seconds': result['seconds'], 'retries': result['retries']}
        target.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        kinds = [e['type'][0] for e in data['elements']]
        print(f"p{page:03d} {result['seconds']:5.1f}s elements={''.join(kinds)} "
              f"tokens={result['usage'].get('totalTokenCount')}", flush=True)


if __name__ == '__main__':
    main()
