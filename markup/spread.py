"""Перенос ручной правки на похожие кадры.

Правка одного кадра чинит один кадр. Но промахи автомата однотипны: если на
кольце в три четверти он обвёл только камень без каста, то же самое он сделал и
на полусотне соседних. Модуль ищет такие кадры и переносит на них ту же поправку.

Как устроен перенос. Правка описывается не абсолютными пикселями, а
преобразованием в собственных координатах элемента: во сколько раз растянуть
вдоль длинной оси, во сколько поперёк, куда сместить центр в долях сторон и на
сколько довернуть. Такое описание не зависит от масштаба кадра, поэтому одна и та
же правка ложится и на крупное кольцо, и на мелкое.

Порядок работы намеренно трёхступенчатый, и человек стоит в конце:

1. **Отбор по признакам** — бесплатно. Тот же тип изделия, тот же ракурс, тот же
   источник рамки, близкие табличные пропорции.
2. **Проверка Опусом** — он смотрит на кадр-образец и на кандидатов и отвечает,
   у кого та же беда. Числа похожесть не доказывают, а глаза доказывают.
3. **Предложение, а не правка.** Подтверждённые кадры перерисовываются и
   помечаются как предложенные; в галерее они отбираются отдельным фильтром.
   Принять или отклонить решает человек.

Молча переписать девятьсот кадров нельзя: одна неверная догадка испортила бы всю
выборку, а заметили бы это уже у заказчика.
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime
from pathlib import Path

from .advisor import AdvisorError, _encode, call, load_rules, parse_json
from .config import API_KEY_ENV

BATCH = 6              # сколько кандидатов показываем Опусу за один запрос
DEFAULT_LIMIT = 12     # сколько кандидатов разбираем по умолчанию
ASPECT_TOL = 1.25      # допустимое расхождение табличных пропорций с образцом

SYSTEM = (
    "Ты — инженер-технолог ювелирного производства. Ты решаешь, к каким фото "
    "применима та же поправка разметки, что уже сделана на образце. "
    "Отвечай строго одним JSON-объектом."
)

PROMPT = """Первое изображение — ОБРАЗЕЦ ({source}). На нём человек поправил рамку
декоративного элемента, потому что автомат ошибся так:

{diagnosis}

Дальше идут кандидаты. По каждому реши, повторяется ли на нём та же ошибка
автомата и подойдёт ли та же поправка.

Кандидаты:
{listing}

Верни JSON:
{{
  "подходят": ["S2-000", ...],
  "не_подходят": ["S2-000", ...],
  "пояснение": "одной фразой, по какому признаку разделил"
}}

В "подходят" клади только те, где ошибка действительно та же. Сомневаешься —
клади в "не_подходят": лишняя правка хуже пропущенной.

Отвечай только JSON."""


# ---------------------------------------------------------------- поправка

def transform(auto: dict, manual: dict) -> dict:
    """Правка в координатах элемента: растяжение, смещение центра, доворот."""
    def достать(rect: dict):
        центр = rect.get("center") or [rect.get("cx", 0.0), rect.get("cy", 0.0)]
        return (float(центр[0]), float(центр[1]), float(rect.get("long", 0.0)),
                float(rect.get("short", 0.0)), float(rect.get("angle", 0.0)))

    acx, acy, along, ashort, aangle = достать(auto)
    mcx, mcy, mlong, mshort, mangle = достать(manual)
    if along < 1 or ashort < 1:
        return {}

    угол = math.radians(aangle)
    ux, uy = math.cos(угол), math.sin(угол)
    nx, ny = -uy, ux
    dx, dy = mcx - acx, mcy - acy
    return {
        "растяжение_вдоль": round(mlong / along, 4),
        "растяжение_поперёк": round(mshort / ashort, 4),
        "сдвиг_вдоль": round((dx * ux + dy * uy) / along, 4),
        "сдвиг_поперёк": round((dx * nx + dy * ny) / ashort, 4),
        "доворот": round(mangle - aangle, 2),
    }


def apply_transform(element, поправка: dict) -> dict:
    """Накладывает поправку на автоматический элемент кандидата."""
    ux, uy = element.axis
    nx, ny = element.normal
    вдоль = поправка.get("сдвиг_вдоль", 0.0) * element.long
    поперёк = поправка.get("сдвиг_поперёк", 0.0) * element.short
    return {
        "center": [float(element.center[0] + ux * вдоль + nx * поперёк),
                   float(element.center[1] + uy * вдоль + ny * поперёк)],
        "long": float(max(12.0, element.long * поправка.get("растяжение_вдоль", 1.0))),
        "short": float(max(12.0, element.short * поправка.get("растяжение_поперёк", 1.0))),
        "angle": float(element.angle + поправка.get("доворот", 0.0)),
    }


# ---------------------------------------------------------------- правки раскладки

def tweaks_of(space, source_id: str) -> dict:
    """Правки разметки образца в безразмерном виде — годном для чужого кадра.

    Записывается то же, что и в журнале привычек: вынос размерной линии от кромки
    силуэта в долях эшелона, полка выноски в долях рамки изделия, сторона полки.
    К ним добавляются концы габарита — в долях рамки декоративного элемента, и
    стрелка выноски — в собственных координатах элемента.

    Интервал шинки не переносится вовсе. Место промера выбирается по самому
    кадру — по тому участку ободка, где лицевая грань видна целиком, — и
    перенести его на другое кольцо значит поставить метку не на металл.
    """
    from . import anchors, habits
    from .layout import compose, find_decor
    from .render import InclinedDim, Leader, LinearDim

    план = space.plan(source_id)
    if not план.tweaks:
        return {}
    row = space.rows[source_id]
    product = space.product(source_id)
    элемент, _ = find_decor(product, план, row)
    лист = compose(product, план, row, getattr(space, "echelons", {}).get(source_id, {}),
                   элемент, space.cache)
    bx0, by0, bx1, by1 = элемент.box
    итог: dict = {}
    for ключ in план.tweaks:
        if ключ not in лист.keys:
            continue
        item = лист.items[лист.keys.index(ключ)]
        правка = план.tweaks.get(ключ) or {}
        запись = habits.отметка(product, item, правка) or {}
        # переносится только то, что человек тронул: концы габарита, записанные
        # «заодно», прибили бы их на кандидате к кромкам его рамки — а рамка у
        # него своя, и правка о ней ничего не говорила
        тронуты_концы = bool({"a", "b"} & set(правка))
        if isinstance(item, LinearDim) and тронуты_концы:
            начало, конец = (bx0, bx1) if item.axis == "h" else (by0, by1)
            сторона = max(конец - начало, 1.0)
            запись["ось"] = item.axis
            запись["концы"] = [round((item.a - начало) / сторона, 4),
                               round((item.b - начало) / сторона, 4)]
        elif isinstance(item, InclinedDim) and тронуты_концы:
            запись["концы_локально"] = [list(anchors.to_local(item.a, элемент)),
                                        list(anchors.to_local(item.b, элемент))]
        elif isinstance(item, Leader) and "anchor" in правка:
            запись["стрелка"] = list(anchors.to_local(item.anchor, элемент))
        if запись:
            итог[ключ] = запись
    return итог


def apply_tweaks(product, элемент, поправки: dict) -> dict:
    """Разворачивает безразмерные правки образца в координаты этого кадра."""
    from . import anchors, habits

    px0, py0, px1, py1 = product.bbox
    bx0, by0, bx1, by1 = элемент.box
    готовые: dict = {}
    for ключ, запись in (поправки or {}).items():
        правка: dict = {}
        вид = запись.get("вид")
        if вид in ("габарит", "наклонный") and "вынос" in запись:
            if запись.get("край"):
                правка["level"] = habits._к_кромке(запись["край"], запись["вынос"],
                                                   product.bbox)
            elif вид == "наклонный":
                правка["offset"] = запись["вынос"] * habits.STYLE.offset_1
        if "концы" in запись and запись.get("ось") in ("h", "v"):
            начало, конец = ((bx0, bx1) if запись["ось"] == "h" else (by0, by1))
            сторона = max(конец - начало, 1.0)
            правка["a"] = начало + запись["концы"][0] * сторона
            правка["b"] = начало + запись["концы"][1] * сторона
        if "концы_локально" in запись:
            правка["a"] = list(anchors.to_frame(запись["концы_локально"][0], элемент))
            правка["b"] = list(anchors.to_frame(запись["концы_локально"][1], элемент))
        if вид == "выноска" and "полка" in запись:
            правка["elbow"] = [px0 + запись["полка"][0] * max(px1 - px0, 1.0),
                               py0 + запись["полка"][1] * max(py1 - py0, 1.0)]
            if "сторона" in запись:
                правка["direction"] = int(запись["сторона"])
        if "стрелка" in запись:
            правка["anchor"] = list(anchors.to_frame(запись["стрелка"], элемент))
        if правка:
            готовые[ключ] = правка
    return готовые


def _significant(поправка: dict) -> bool:
    return (bool(поправка.get("поправки"))
            or abs(поправка.get("растяжение_вдоль", 1) - 1) > 0.08
            or abs(поправка.get("растяжение_поперёк", 1) - 1) > 0.08
            or abs(поправка.get("сдвиг_вдоль", 0)) > 0.06
            or abs(поправка.get("сдвиг_поперёк", 0)) > 0.06
            or abs(поправка.get("доворот", 0)) > 4)


# ---------------------------------------------------------------- отбор

def candidates(space, source_id: str, limit: int = DEFAULT_LIMIT) -> list[str]:
    """Кадры, на которых автомат мог ошибиться так же. Только по признакам, без модели."""
    образец = space.plan(source_id)
    строка = space.rows[source_id]
    эталон = _aspect(строка)

    отобранные: list[tuple[float, str]] = []
    for item_id in space.order:
        if item_id == source_id:
            continue
        путь = space.plan_path(item_id)
        if not путь.exists():
            continue          # без готового плана кандидат стоил бы обращения к модели
        try:
            данные = json.loads(путь.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if данные.get("manual") or данные.get("proposed"):
            continue
        if данные.get("item_type") != образец.item_type:
            continue
        if данные.get("view") != образец.view:
            continue
        доля = _aspect(space.rows[item_id])
        if эталон > 0 and доля > 0:
            расхождение = abs(math.log(доля / эталон))
            if расхождение > math.log(ASPECT_TOL):
                continue
        else:
            расхождение = 1.0
        отобранные.append((расхождение, item_id))

    отобранные.sort()
    return [item_id for _, item_id in отобранные[:limit]]


def _aspect(row) -> float:
    длина, ширина = max(row.length, row.width), min(row.length, row.width)
    return длина / ширина if ширина > 0 else 0.0


# ---------------------------------------------------------------- проверка Опусом

def _diagnosis(space, source_id: str, поправка: dict) -> str:
    """Словесное описание правки: берём разбор Опуса, если он есть, иначе числа."""
    for запись in reversed(load_rules(space.cache)):
        if запись.get("id") == source_id and запись.get("причина"):
            return f"{запись['причина']}\nПравило: {запись.get('правило', '')}"
    части = []
    for строка in поправка.get("словами") or []:
        части.append(строка.lstrip("- "))
    if поправка.get("растяжение_вдоль", 1) != 1:
        части.append(f"рамку растянули вдоль оси в {поправка['растяжение_вдоль']:.2f} раза")
    if поправка.get("растяжение_поперёк", 1) != 1:
        части.append(f"поперёк — в {поправка['растяжение_поперёк']:.2f} раза")
    if abs(поправка.get("доворот", 0)) > 1:
        части.append(f"довернули на {поправка['доворот']:+.0f}°")
    return "Человек поправил рамку: " + ", ".join(части) if части else "Рамку слегка поправили."


def verify(space, source_id: str, ids: list[str], поправка: dict) -> dict:
    """Спрашивает Опуса, к каким кандидатам применима та же правка."""
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        return {"ошибка": f"не задан ключ {API_KEY_ENV}"}

    подходят: list[str] = []
    пояснения: list[str] = []
    diagnosis = _diagnosis(space, source_id, поправка)

    for начало in range(0, len(ids), BATCH):
        пачка = ids[начало:начало + BATCH]
        listing = "\n".join(
            f"{номер}. {item_id} — {space.rows[item_id].article}"
            for номер, item_id in enumerate(пачка, 1))
        содержимое = [
            {"type": "text", "text": PROMPT.format(
                source=source_id, diagnosis=diagnosis, listing=listing)},
            {"type": "text", "text": f"ОБРАЗЕЦ {source_id}:"},
            _image(space, source_id),
        ]
        for номер, item_id in enumerate(пачка, 1):
            содержимое.append({"type": "text", "text": f"КАНДИДАТ {номер}: {item_id}"})
            содержимое.append(_image(space, item_id))
        try:
            ответ = _ask(содержимое, api_key)
        except (AdvisorError, ValueError, OSError) as exc:
            return {"ошибка": str(exc), "подходят": подходят}
        подходят += [i for i in ответ.get("подходят") or [] if i in пачка]
        if ответ.get("пояснение"):
            пояснения.append(str(ответ["пояснение"]))

    return {"подходят": подходят, "пояснение": " ".join(пояснения)}


def _image(space, item_id: str) -> dict:
    product = space.product(item_id)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": _encode(product.canvas)}}


def _ask(content: list, api_key: str, timeout: int = 600) -> dict:
    return parse_json(call(content, api_key, system=SYSTEM, timeout=timeout))


# ---------------------------------------------------------------- применение

def propose(space, ids: list[str], поправка: dict, source_id: str) -> list[dict]:
    """Накладывает поправку на кандидатов и перерисовывает их как предложенные."""
    from .layout import find_decor

    правит_рамку = any(ключ in поправка for ключ in
                       ("растяжение_вдоль", "растяжение_поперёк", "сдвиг_вдоль",
                        "сдвиг_поперёк", "доворот"))
    сделано = []
    for item_id in ids:
        try:
            план = space.plan(item_id)
            product = space.product(item_id)
            элемент, _ = find_decor(product, план, space.rows[item_id])
            if правит_рамку:
                rect = apply_transform(элемент, поправка)
                план.decor_rect = {"center": rect["center"], "long": rect["long"],
                                   "short": rect["short"], "angle": rect["angle"]}
                план.decor_lock = "oriented"
                половина = max(rect["long"], rect["short"]) / 2
                план.decor_box = (float(rect["center"][0] - половина),
                                  float(rect["center"][1] - половина),
                                  float(rect["center"][0] + половина),
                                  float(rect["center"][1] + половина))
                # концы габарита отмеряются от рамки, поэтому раскладку кладём на
                # уже поправленный элемент, а не на тот, что был до переноса
                элемент, _ = find_decor(product, план, space.rows[item_id])
            if поправка.get("поправки"):
                план.tweaks = {**план.tweaks,
                               **apply_tweaks(product, элемент, поправка["поправки"])}
            план.proposed = source_id
            space.save_plan(item_id, план)
            _путь, замечания = space.render(item_id)
            сделано.append({"id": item_id, "замечания": замечания})
        except Exception as exc:                     # noqa: BLE001 — один кадр не должен рушить пачку
            сделано.append({"id": item_id, "ошибка": str(exc)})
    _журнал(space.cache, source_id, поправка, [с["id"] for с in сделано])
    return сделано


def _журнал(cache: Path, source_id: str, поправка: dict, ids: list[str]) -> None:
    путь = cache / "переносы.json"
    записи = []
    if путь.exists():
        try:
            записи = json.loads(путь.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            записи = []
    записи.append({"образец": source_id, "поправка": поправка, "кадры": ids,
                   "когда": datetime.now().isoformat(timespec="seconds")})
    путь.parent.mkdir(parents=True, exist_ok=True)
    путь.write_text(json.dumps(записи, ensure_ascii=False, indent=2), encoding="utf-8")


def run(space, source_id: str, limit: int = DEFAULT_LIMIT, verify_with_model: bool = True) -> dict:
    """Полный проход: отбор, проверка, предложение."""
    план = space.plan(source_id)
    if not (план.manual and (план.decor_rect or план.tweaks)):
        return {"ошибка": "у образца нет ручной правки — переносить нечего"}

    from .layout import find_decor
    from .planner import Plan

    product = space.product(source_id)
    поправка: dict = {}
    if план.decor_rect:
        свободный = Plan.from_dict(план.to_dict())
        свободный.decor_lock, свободный.decor_rect, свободный.manual = "", None, False
        авто, _ = find_decor(product, свободный, space.rows[source_id])
        поправка = transform(
            {"center": авто.center, "long": авто.long, "short": авто.short,
             "angle": авто.angle}, план.decor_rect)
    # правка раскладки переносится наравне с рамкой: «полка не с той стороны»
    # повторяется на соседних кадрах ровно так же, как промах по рамке
    правки = tweaks_of(space, source_id)
    if правки:
        from .advisor import moves

        поправка["поправки"] = правки
        # словами — чтобы Опус на проверке видел, что именно переносится, а
        # человек в журнале переносов мог прочитать это без пересчёта долей
        поправка["словами"] = moves(space, source_id, план)
    if not поправка:
        return {"ошибка": "не удалось описать правку"}
    if not _significant(поправка):
        return {"ошибка": "правка слишком мелкая, переносить нечего", "поправка": поправка}

    отобранные = candidates(space, source_id, limit)
    if not отобранные:
        return {"поправка": поправка, "кандидаты": [], "подходят": [],
                "пояснение": "похожих кадров с готовым планом не нашлось"}

    итог = {"поправка": поправка, "кандидаты": отобранные}
    if verify_with_model:
        проверка = verify(space, source_id, отобранные, поправка)
        итог.update(проверка)
    else:
        итог["подходят"] = отобранные
        итог["пояснение"] = "проверка моделью отключена, отбор только по признакам"
    return итог
