"""Журнал привычек раскладки: что человек переставляет одинаково от кадра к кадру.

Правка мышью чинит один кадр. Ценность появляется, когда из десятка правок видно
привычку — и автомат начинает делать так же сам. Это тот же приём, что уже
работает на якоре выноски высоты (`markup/anchors.py`), только распространённый на
всю разметку: размерные линии, полки выносок и сторону, в которую полка смотрит.

Привычкой считается не всякая правка. Записывается только то, что человек делает
одинаково от кадра к кадру и что не зависит от масштаба:

- **вынос размерной линии** — на каком расстоянии от изделия человек держит
  габарит, в долях первого эшелона (`STYLE.offset_1`), и от какой кромки силуэта
  он его отмеряет;
- **полка выноски** — куда он ставит излом, в долях рамки изделия;
- **сторона полки** — влево или вправо она смотрит.

А вот **концы габарита привычкой не являются**. То, докуда тянется декоративный
элемент, — свойство конкретного кадра, а не привычка технолога: перенести его на
соседнее кольцо значит соврать про его размер. Такие правки идут другим путём —
через разбор Опусом (`markup/advisor.py`) и перенос на кадры, подтверждённые
человеком (`markup/spread.py`).

Порог тот же, что у якоря: три отметки на пару «тип изделия + ракурс», вдвое
больше — чтобы перенести привычку на другой ракурс того же типа. Берётся медиана
(для стороны полки — большинство): один промах мышью её не сдвинет.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .anchors import _median
from .config import STYLE

MIN_SAMPLES = 3        # с какого числа отметок доверяем медиане
FILE = "привычки.json"

#: Границы правдоподобия. Вынос за ними — не привычка, а промах мышью или кадр,
#: где раскладка разъехалась; в медиану такие отметки не идут.
REACH = (0.15, 6.0)          # вынос размерной линии, в долях первого эшелона
SHELF = (-1.5, 2.5)          # полка выноски, в долях рамки изделия


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


def _save(cache: Path, записи: list[dict]) -> None:
    target = path(cache)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(записи, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- запись

def _от_кромки(level: float, axis: str, bbox) -> tuple[str, float]:
    """Вынос размерной линии от ближней кромки силуэта, в долях первого эшелона."""
    px0, py0, px1, py1 = bbox
    низ, верх = (py1, py0) if axis == "h" else (px1, px0)
    имена = ("низ", "верх") if axis == "h" else ("право", "лево")
    if abs(level - верх) <= abs(level - низ):
        return имена[1], (верх - level) / STYLE.offset_1
    return имена[0], (level - низ) / STYLE.offset_1


def _к_кромке(край: str, вынос: float, bbox) -> float:
    px0, py0, px1, py1 = bbox
    return {"верх": py0 - вынос * STYLE.offset_1,
            "низ": py1 + вынос * STYLE.offset_1,
            "лево": px0 - вынос * STYLE.offset_1,
            "право": px1 + вынос * STYLE.offset_1}[край]


def отметка(product, item, поправка: dict | None = None) -> dict | None:
    """Переводит поправленный элемент разметки в запись журнала.

    Записывается только то, что человек действительно тронул. Иначе правка конца
    габарита заодно «подтверждала» бы вынос размерной линии, которого никто не
    двигал, — и через три кадра автомат намертво приколотил бы линию туда, куда
    он и сам её ставил, отняв у раскладки право развести подписи эшелонами.

    Возвращает None, когда привычке взяться не из чего: тронуты только концы
    габарита или интервал шинки — они про этот кадр, а не про руку человека.
    """
    from .render import InclinedDim, Leader, LinearDim

    тронуто = set(поправка or {}) or {"level", "offset", "elbow", "direction"}
    bbox = product.bbox
    if isinstance(item, LinearDim):
        if "level" not in тронуто:
            return None
        край, вынос = _от_кромки(item.level, item.axis, bbox)
        if not REACH[0] <= вынос <= REACH[1]:
            return None
        return {"вид": "габарит", "ось": item.axis, "край": край, "вынос": round(вынос, 3)}
    if isinstance(item, InclinedDim):
        if "offset" not in тронуто:
            return None
        вынос = item.offset / STYLE.offset_1
        if not REACH[0] <= вынос <= REACH[1]:
            return None
        return {"вид": "наклонный", "вынос": round(вынос, 3)}
    if isinstance(item, Leader):
        if not тронуто & {"elbow", "direction"}:
            return None
        px0, py0, px1, py1 = bbox
        ширина, высота = max(px1 - px0, 1.0), max(py1 - py0, 1.0)
        полка = [(item.elbow[0] - px0) / ширина, (item.elbow[1] - py0) / высота]
        if not all(SHELF[0] <= доля <= SHELF[1] for доля in полка):
            return None
        return {"вид": "выноска", "полка": [round(доля, 3) for доля in полка],
                "сторона": int(item.direction)}
    return None


def remember(cache: Path, item_id: str, plan, product, sheet, ключи) -> int:
    """Записывает привычки по тем размерам, которые человек только что тронул."""
    записи = load(cache)
    добавлено = 0
    for индекс, item in enumerate(sheet.items):
        ключ = sheet.key_at(индекс)
        if ключ not in ключи:
            continue
        запись = отметка(product, item, (plan.tweaks or {}).get(ключ))
        if not запись:
            continue
        записи = [r for r in записи
                  if not (r.get("id") == item_id and r.get("размер") == ключ)]
        записи.append({"id": item_id, "тип": plan.item_type or "прочее",
                       "ракурс": plan.view or "фас", "размер": ключ, **запись,
                       "когда": datetime.now().isoformat(timespec="seconds")})
        добавлено += 1
    if добавлено:
        _save(cache, записи)
    return добавлено


def forget(cache: Path, item_id: str, ключи=None) -> None:
    """Убирает отметки кадра: правку сняли — привычку она больше не подтверждает."""
    записи = [r for r in load(cache)
              if r.get("id") != item_id
              or (ключи is not None and r.get("размер") not in ключи)]
    _save(cache, записи)


# ---------------------------------------------------------------- чтение

def _большинство(значения: list) -> tuple:
    счёт: dict = {}
    for значение in значения:
        ключ = tuple(значение) if isinstance(значение, list) else значение
        счёт[ключ] = счёт.get(ключ, 0) + 1
    return max(счёт.items(), key=lambda пара: пара[1])


def _свод(группа: list[dict]) -> dict:
    """Медиана по числам и большинство по стороне полки для одной группы отметок."""
    вид = _большинство([r.get("вид") for r in группа])[0]
    свои = [r for r in группа if r.get("вид") == вид]
    итог: dict = {"вид": вид, "отметок": len(свои)}
    if вид in ("габарит", "наклонный"):
        итог["вынос"] = round(_median([r["вынос"] for r in свои]), 3)
        оси = [r.get("ось") for r in свои if r.get("ось")]
        if оси:
            итог["ось"] = _большинство(оси)[0]
        края = [r.get("край") for r in свои if r.get("край")]
        if края:
            итог["край"] = _большинство(края)[0]
    elif вид == "выноска":
        итог["полка"] = [round(_median([r["полка"][0] for r in свои]), 3),
                         round(_median([r["полка"][1] for r in свои]), 3)]
        сторона, сколько = _большинство([r["сторона"] for r in свои])
        # сторона полки — выбор из двух, и меняет она кадр сильно: разворот уводит
        # подпись через весь холст. Берём её, только когда человек разворачивал
        # полку одинаково почти всегда, а не «чаще, чем нет»
        if сколько >= max(MIN_SAMPLES, len(свои) * 0.75):
            итог["сторона"] = int(сторона)
    return итог


def learned(cache: Path, item_type: str, view: str) -> dict:
    """Привычки по этой паре «тип + ракурс»: {размер: свод}, если их уже набрали."""
    записи = load(cache)
    итог: dict = {}
    for отбор, порог in (
            (lambda r: r.get("тип") == item_type and r.get("ракурс") == view, MIN_SAMPLES),
            (lambda r: r.get("тип") == item_type, MIN_SAMPLES * 2),
    ):
        группы: dict[str, list] = {}
        for r in (r for r in записи if отбор(r)):
            группы.setdefault(r.get("размер"), []).append(r)
        for размер, группа in группы.items():
            if размер not in итог and len(группа) >= порог:
                итог[размер] = _свод(группа)
    return итог


def resolve(cache, plan, product) -> dict:
    """Привычки в виде поправок для этого кадра: {размер: {'level': ...}}.

    Возвращается то же самое, чем описывается ручная правка, — раскладка
    накладывает обе одинаково. Ручная правка при совпадении главнее: человек на
    этом кадре сказал больше, чем его же привычка вообще.
    """
    if cache is None:
        return {}
    свод = learned(Path(cache), plan.item_type or "прочее", plan.view or "фас")
    поправки: dict = {}
    for размер, привычка in свод.items():
        if привычка["отметок"] < MIN_SAMPLES:
            continue
        if привычка["вид"] == "габарит":
            ось = привычка.get("ось")
            свои = {plan.axes.get("horizontal"): "h", plan.axes.get("vertical"): "v"}
            # привычка снята с горизонтального габарита — на кадре, где тот же
            # размер лёг на вертикаль, она означала бы совсем другое место
            if ось and свои.get(размер) != ось:
                continue
            if привычка.get("край"):
                поправки[размер] = {
                    "level": _к_кромке(привычка["край"], привычка["вынос"], product.bbox)}
        elif привычка["вид"] == "наклонный":
            поправки[размер] = {"offset": привычка["вынос"] * STYLE.offset_1}
        elif привычка["вид"] == "выноска":
            px0, py0, px1, py1 = product.bbox
            поправка = {"elbow": [px0 + привычка["полка"][0] * (px1 - px0),
                                  py0 + привычка["полка"][1] * (py1 - py0)]}
            if "сторона" in привычка:
                поправка["direction"] = привычка["сторона"]
            поправки[размер] = поправка
    return поправки


def _словами(свод: dict) -> str:
    if свод["вид"] in ("габарит", "наклонный"):
        куда = свод.get("край", "от оси элемента")
        return f"{куда}, вынос {свод['вынос']:.2f} эшелона"
    полка = свод.get("полка") or [0, 0]
    сторона = свод.get("сторона")
    хвост = "" if сторона is None else (", полка влево" if сторона < 0 else ", полка вправо")
    return f"излом {полка[0]:.2f} / {полка[1]:.2f} рамки{хвост}"


def summary(cache: Path) -> list[dict]:
    """Сводка для панели «Чему научился»: что набрано и работает ли оно."""
    группы: dict[tuple, list] = {}
    for r in load(cache):
        группы.setdefault((r.get("тип"), r.get("ракурс"), r.get("размер")), []).append(r)
    итог = []
    for (тип, ракурс, размер), группа in sorted(группы.items(), key=lambda г: -len(г[1])):
        свод = _свод(группа)
        итог.append({"тип": тип, "ракурс": ракурс, "размер": размер,
                     "отметок": len(группа), "вид": свод["вид"],
                     "значение": _словами(свод),
                     "работает": len(группа) >= MIN_SAMPLES})
    return итог
