"""Журнал ручных отметок якоря выноски высоты и обучение на них.

Выноска высоты — самое спорное место разметки: высоты на ракурсе не видно, и
куда именно поставить стрелку, решает привычка технолога, а не геометрия. Автомат
берёт ближайшую к полке точку металла внутри декоративного элемента; это никогда
не даёт стрелку в пустоту, но не всегда попадает туда, куда поставил бы человек.

Поэтому каждая ручная отметка запоминается в собственных координатах элемента:
доля вдоль длинной оси и доля поперёк неё, обе в диапазоне [-1, 1]. Как только по
паре «тип изделия + ракурс» набирается несколько отметок, автомат берёт их медиану
и ставит якорь туда же. Медиана устойчива к одному промаху — а промахи мышью
случаются.

Это и есть всё «обучение»: никакого дообучения модели, простая калибровка по
показаниям человека, которую видно глазами в `cache/якоря.json`.
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

MIN_SAMPLES = 3        # с какого числа отметок доверяем медиане
FILE = "якоря.json"


def path(cache: Path) -> Path:
    return cache / FILE


def load(cache: Path) -> list[dict]:
    target = path(cache)
    if not target.exists():
        return []
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def to_local(point, element) -> tuple[float, float]:
    """Точка кадра в координатах элемента: доли вдоль оси и поперёк неё."""
    ux, uy = element.axis
    nx, ny = element.normal
    dx, dy = point[0] - element.center[0], point[1] - element.center[1]
    half_long = max(element.long / 2, 1e-6)
    half_short = max(element.short / 2, 1e-6)
    return ((dx * ux + dy * uy) / half_long, (dx * nx + dy * ny) / half_short)


def to_frame(local, element) -> tuple[float, float]:
    """Обратный перевод: доли элемента в координаты кадра."""
    ux, uy = element.axis
    nx, ny = element.normal
    along, across = local
    return (element.center[0] + ux * along * element.long / 2 + nx * across * element.short / 2,
            element.center[1] + uy * along * element.long / 2 + ny * across * element.short / 2)


def remember(cache: Path, item_id: str, item_type: str, view: str,
             point, element) -> None:
    """Записывает отметку человека в журнал."""
    along, across = to_local(point, element)
    записи = [r for r in load(cache) if r.get("id") != item_id]
    записи.append({
        "id": item_id, "тип": item_type or "прочее", "ракурс": view or "фас",
        "вдоль": round(along, 3), "поперёк": round(across, 3),
        "когда": datetime.now().isoformat(timespec="seconds"),
    })
    target = path(cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(записи, ensure_ascii=False, indent=2), encoding="utf-8")


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def learned(cache: Path, item_type: str, view: str) -> tuple[float, float] | None:
    """Медианное положение якоря для этой пары «тип + ракурс», если оно набрано.

    Сначала ищем отметки для того же ракурса, затем — для того же типа изделия
    вообще: ракурс уточняет привычку, но и без него отметки лучше умолчания.
    """
    записи = load(cache)
    # Для своего ракурса хватает трёх отметок. Переносить привычку на другие
    # ракурсы того же типа изделия рискованнее — там нужен вдвое больший запас.
    for отбор, порог in (
            (lambda r: r.get("тип") == item_type and r.get("ракурс") == view, MIN_SAMPLES),
            (lambda r: r.get("тип") == item_type, MIN_SAMPLES * 2),
    ):
        подходящие = [r for r in записи if отбор(r)]
        if len(подходящие) >= порог:
            return (_median([r["вдоль"] for r in подходящие]),
                    _median([r["поперёк"] for r in подходящие]))
    return None


def summary(cache: Path) -> list[dict]:
    """Сводка по журналу — что уже выучено и на скольких отметках."""
    записи = load(cache)
    группы: dict[tuple, list] = {}
    for r in записи:
        группы.setdefault((r.get("тип"), r.get("ракурс")), []).append(r)
    итог = []
    for (тип, ракурс), список in sorted(группы.items(), key=lambda g: -len(g[1])):
        итог.append({
            "тип": тип, "ракурс": ракурс, "отметок": len(список),
            "вдоль": round(_median([r["вдоль"] for r in список]), 2),
            "поперёк": round(_median([r["поперёк"] for r in список]), 2),
            "работает": len(список) >= MIN_SAMPLES,
        })
    return итог
