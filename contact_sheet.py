#!/usr/bin/env python
"""Контактный лист: собирает готовые PNG в сетку для быстрой визуальной приёмки.

    python contact_sheet.py --dir out --out out/_контроль/пилот.png --cols 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from markup.render import font


def build(paths: list[Path], cols: int, cell: int, label_h: int) -> Image.Image:
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    f = font("caption", max(12, label_h - 8))

    for i, path in enumerate(paths):
        x = (i % cols) * cell
        y = (i // cols) * (cell + label_h)
        tile = Image.open(path).convert("RGB")
        tile.thumbnail((cell - 8, cell - 8), Image.LANCZOS)
        sheet.paste(tile, (x + (cell - tile.width) // 2, y + (cell - tile.height) // 2))
        draw.text((x + 6, y + cell + 2), path.stem, font=f, fill=(30, 30, 30))
    return sheet


def main() -> int:
    p = argparse.ArgumentParser(description="Контактный лист по готовым PNG")
    p.add_argument("--dir", default="out", help="каталог с результатами")
    p.add_argument("--out", default="out/_контроль/контактный_лист.png")
    p.add_argument("--cols", type=int, default=5)
    p.add_argument("--cell", type=int, default=400, help="сторона ячейки, px")
    p.add_argument("--limit", type=int, help="взять только первые N файлов")
    args = p.parse_args()

    paths = sorted(q for q in Path(args.dir).glob("*.png") if not q.name.startswith("_"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        print("В каталоге нет готовых PNG.")
        return 1

    sheet = build(paths, args.cols, args.cell, label_h=26)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(f"{out} — {len(paths)} изображений, {sheet.width}x{sheet.height}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
