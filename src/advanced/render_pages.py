"""Render every PDF page to PNG (for the vision model and for human verification).

python src/advanced/render_pages.py --pdf starter/data/advanced/vn-ccdr-2022.pdf --out starter/data/advanced/pages --dpi 110
"""
import argparse
from pathlib import Path

import pymupdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dpi', type=int, default=110)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with pymupdf.open(args.pdf) as doc:
        for index, page in enumerate(doc, start=1):
            page.get_pixmap(dpi=args.dpi).save(args.out / f'p{index:03d}.png')
        print(f'{len(doc)} pages -> {args.out}')


if __name__ == '__main__':
    main()
