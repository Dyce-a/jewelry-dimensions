"""Разбор кадра для веб-редактора: что скрипт увидел и что из этого нарисовал.

Модуль ничего не решает сам — он только достаёт наружу ту геометрию, по которой
раскладка уже приняла свои решения. Поэтому редактор показывает не пересказ, а
ровно то, что легло в PNG: те же рамки, те же якоря, те же промеры.
"""
from __future__ import annotations

import math
from pathlib import Path

from .config import CANVAS, DIMENSIONS, STYLE
from .imaging import Product, fetch, prepare
from .layout import (_band_column, _band_section, _bottom_run, _cut, _flank_point,
                     _mm_per_px, find_decor)
from .planner import Plan
from .render import TEXT_PAD, InclinedDim, Leader, LinearDim

PREVIEW = 1000     # сторона превью холста для редактора, px
CAPTIONS = dict(DIMENSIONS)


def canvas_preview(product: Product, path: Path) -> Path:
    """Кладёт рядом уменьшенный кадр без разметки — подложку редактора."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        preview = product.canvas.copy()
        preview.thumbnail((PREVIEW, PREVIEW))
        preview.save(path, format="JPEG", quality=90)
    return path


def _plain(value):
    """numpy-числа в обычные python-числа: их не берёт json."""
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if hasattr(value, "item"):
        return value.item()
    return value


def _corners(element) -> list[list[float]]:
    ux, uy = element.axis
    nx, ny = element.normal
    half_long, half_short = element.long / 2, element.short / 2
    return [[element.center[0] + sx * ux * half_long + sy * nx * half_short,
             element.center[1] + sx * uy * half_long + sy * ny * half_short]
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _shapes(sheet) -> list[dict]:
    """Нарисованные элементы разметки в координатах холста.

    Каждая фигура несёт свой размер из таблицы и отметку «правлено». Без них
    редактор не знал бы, за какой размер он тянет: очередь фигур непостоянна —
    размер, не сошедшийся с кадром, уходит с габарита на выноску и меняет место
    в списке.
    """
    shapes: list[dict] = []
    for индекс, item in enumerate(sheet.items):
        ключ = sheet.key_at(индекс)
        общее = {"key": ключ, "caption": CAPTIONS.get(ключ, ""),
                 "tweaked": ключ in sheet.tweaked}
        if isinstance(item, LinearDim):
            shapes.append({**общее, "kind": "linear", "axis": item.axis,
                           "a": item.a, "b": item.b, "base": item.base,
                           "level": item.level, "side": item.side, "text": item.text,
                           "box": list(item.box) if item.box else None})
        elif isinstance(item, InclinedDim):
            shapes.append({**общее, "kind": "inclined", "a": list(item.a),
                           "b": list(item.b), "normal": list(item.normal),
                           "offset": item.offset, "text": item.text})
        elif isinstance(item, Leader):
            shapes.append({**общее, "kind": "leader", "anchor": list(item.anchor),
                           "elbow": list(item.elbow), "direction": item.direction,
                           "length": item.length, "arrow": item.arrow,
                           "text": item.text, "label": item.caption,
                           "span": [list(p) for p in item.span] if item.span else None})
    return shapes


def describe(row, plan: Plan, product: Product, cache: Path | None = None,
             levels: dict | None = None) -> dict:
    """Полный разбор позиции: элемент, масштаб, промер шинки, нарисованные линии.

    `levels` — эшелоны, на которых остановилась раскладка при последней отрисовке.
    Без них схема показывала бы первый эшелон, а в PNG размерная линия стоит на
    третьем: человек тянул бы мышью не ту линию, которую видит на готовом кадре.
    """
    from .layout import compose

    values = row.dims()
    element, notes = find_decor(product, plan, row)
    sheet = compose(product, plan, row, levels or {}, element, cache)

    mm_per_px = _mm_per_px(plan, values, element)
    by_axis = {}
    if element.short > 0:
        long_mm = max(values["length"], values["width"])
        short_mm = min(values["length"], values["width"])
        by_axis = {"вдоль длинной стороны": long_mm / element.long,
                   "вдоль короткой стороны": short_mm / element.short}
    mismatch = 0.0
    if len(by_axis) == 2:
        a, b = by_axis.values()
        mismatch = abs(math.log(a / b)) if a > 0 and b > 0 else 0.0

    # промеры те же, по которым раскладка строит метки шинки: ширина — сечение в
    # точке, где кадр показывает лицевую грань целиком, толщина — на нижней дуге,
    # куда уходит её выноска
    px0, py0, px1, py1 = product.bbox
    band_col = _band_column(product, plan.band_point)
    run = _bottom_run(product, band_col)
    bottom = (band_col, (run[0] + run[1]) / 2 if run else py1)
    flank = _flank_point(product, element.box,
                         values["band_w"] / mm_per_px if mm_per_px else 0.0)
    width_section = _band_section(product, flank) if flank else None
    thickness_section = _band_section(product, bottom)
    axial = _cut(product, bottom, plan.band_cuts.get("band_w", "v"))
    expected = values["band_w"] / mm_per_px if mm_per_px else 0.0

    def _span(section) -> float:
        return round(math.dist(*section), 1) if section else 0.0

    return _plain({
        "canvas": product.size,
        "product_bbox": list(product.bbox),
        "decor": {
            "box": [round(v, 1) for v in element.box],
            "center": [round(v, 1) for v in element.center],
            "long": round(element.long, 1), "short": round(element.short, 1),
            "angle": round(element.angle, 1), "source": element.source,
            "inclined": element.inclined, "corners": _corners(element),
        },
        "band": {
            "point": list(plan.band_point) if plan.band_point else None,
            "column": band_col,
            "section": [[round(v, 1) for v in p] for p in width_section]
                       if width_section else None,
            "axial_section": [round(v, 1) for v in axial] if axial else None,
            "expected_px": round(expected, 1),
            "measured_px": _span(width_section),
            "thickness_px": _span(thickness_section),
            "visible": plan.band_width_visible,
        },
        "scale": {
            "mm_per_px": round(mm_per_px, 5) if mm_per_px else 0.0,
            "by_axis": {k: round(v, 5) for k, v in by_axis.items()},
            "mismatch": round(mismatch, 3),
        },
        "leader": {
            "anchor": next((s["anchor"] for s in _shapes(sheet)
                            if s["kind"] == "leader"), None),
            "source": sheet.anchor_source,
            "locked": bool(plan.leader_lock and plan.leader_anchor),
            "dim": plan.leader_dim,
        },
        "shapes": _shapes(sheet),
        # размеры стиля отдаём наружу: схема и анимация нанесения рисуют теми же
        # мерками, что и PNG, — иначе на кадре и на схеме окажутся разные линии
        "style": {"label": STYLE.label_size, "caption": STYLE.caption_size,
                  "line": STYLE.line_width, "offset": STYLE.offset_1,
                  "leader": STYLE.leader_len, "tick": STYLE.tick_over,
                  "ext": STYLE.ext_max, "gap": STYLE.ext_gap,
                  "break": STYLE.line_break, "band_tick": STYLE.band_tick,
                  "dot": STYLE.dot_radius, "margin": STYLE.min_margin,
                  "pad": TEXT_PAD, "canvas": product.size},
        # что правлено руками на этом кадре и что автомат подставил из привычек:
        # редактор показывает это разными метками, чтобы человек видел, где он
        # распоряжался сам, а где за него решил накопленный журнал
        "tweaks": dict(plan.tweaks or {}),
        "habits": _habits(plan, product, cache),
        "notes": list(plan.notes) + notes + list(sheet.notes),
    })


def _habits(plan: Plan, product: Product, cache: Path | None) -> dict:
    from . import habits

    try:
        выученное = habits.resolve(cache, plan, product)
    except Exception:                        # noqa: BLE001 — журнал необязателен
        return {}
    return {ключ: список for ключ, список in выученное.items()
            if ключ not in (plan.tweaks or {})}


def load(row, cache_dir: Path, plan_path: Path, canvas: int = CANVAS):
    """Готовит изделие и план позиции; план берётся из кэша, если он там есть."""
    import json

    photo = fetch(row.urls, cache_dir / "photos", row.id)
    product = prepare(photo, canvas=canvas)
    plan = None
    if plan_path.exists():
        try:
            plan = Plan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
            plan = plan.rescale(product.size)
        except (OSError, ValueError, KeyError, TypeError):
            plan = None
    return product, plan
