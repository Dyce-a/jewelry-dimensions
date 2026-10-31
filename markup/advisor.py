"""Разбор ручной правки: почему автомат промахнулся и как не промахиваться дальше.

Правка мышью чинит один кадр. Ценность у неё появляется, когда из неё удаётся
вытащить правило, годное для остальных девятисот. Этим и занят модуль: он
показывает Опусу исходный кадр и то, чем правка человека отличается от
сегодняшнего автомата — рамку против рамки и перестановку разметки, описанную
словами, — и просит назвать причину расхождения в структурированном виде.

Что с этим происходит дальше:

- причина копится в `cache/правила.json` — это журнал, который можно читать
  глазами и который прямо показывает, на чём автомат систематически ошибается;
- сформулированные правила подмешиваются в промпты как few-shot примеры. Правила
  помечены областью — «рамка» или «раскладка», — и подсказке про каст и шинку
  нечего делать в разговоре про вынос размерной линии.

Привычки, которые видны из правок без всякой модели — вынос линии, полка выноски,
сторона полки, — считает `markup/habits.py`; сюда попадает только то, что нужно
объяснить словами.

Дообучения модели тут нет и быть не может: на десятках правок оно бессмысленно,
а через прокси недоступно. Зато накопленные правки — это ещё и эталонный набор,
на котором меряется качество после каждой правки кода.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
from datetime import datetime
from pathlib import Path

from PIL import Image

from .config import API_KEY_ENV, DIMENSIONS

CAPTIONS = dict(DIMENSIONS)

ANTHROPIC_BASE = os.environ.get("PROXYAPI_ANTHROPIC_BASE",
                               "https://api.proxyapi.ru/anthropic/v1")
ADVISOR_MODEL = os.environ.get("ADVISOR_MODEL", "claude-opus-5")
VISION_SIDE = 1024

SYSTEM = (
    "Ты — инженер-технолог ювелирного производства. Ты разбираешь, почему "
    "автоматический поиск декоративного элемента на фото разошёлся с разметкой, "
    "которую поставил человек. Отвечай строго одним JSON-объектом."
)

HEAD = """На фото ювелирное изделие: {item_type}, ракурс «{view}».
По рабочей таблице декоративный элемент — {length} x {width} мм, высота {height} мм,
шинка: толщина {band_t} мм, ширина {band_w} мм.
Координаты — пиксели холста {canvas}x{canvas}, начало в левом верхнем углу, ось Y вниз."""

RECT = """Автомат обвёл декоративный элемент так:
  центр {auto_center}, длина стороны {auto_long}, короткая сторона {auto_short}, наклон {auto_angle}°
  источник рамки: {auto_source}

Человек поправил рамку так:
  центр {manual_center}, длина стороны {manual_long}, короткая сторона {manual_short}, наклон {manual_angle}°"""

MOVES = """Человек переставил на кадре саму разметку:
{список}

Числа при этом не менялись и меняться не могут: они берутся из рабочей таблицы.
Правка говорит о том, где линия должна стоять, а не о том, сколько там миллиметров."""

TAIL = """Посмотри на кадр и объясни расхождение.

Верни JSON:
{{
  "причина": "одной фразой, в чём именно автомат ошибся",
  "правило": "формулировка в повелительном наклонении, годная как подсказка на других кадрах такого же типа; без упоминания этого конкретного артикула",
  "тип_ошибки": "{виды}",
  "масштаб_ошибки": "мелкая|заметная|грубая",
  "обобщается": true|false
}}

"обобщается" ставь false, если промах вызван особенностью именно этого кадра
и правило другим кадрам не поможет.

Отвечай только JSON."""

ВИДЫ_РАМКИ = ("захвачена шинка|захвачена только вставка без каста|перепутаны оси|"
              "захвачен не тот элемент|наклон не учтён|прочее")
ВИДЫ_РАСКЛАДКИ = ("габарит не доведён до конца элемента|габарит заехал на соседнюю деталь|"
                  "размерная линия слишком близко к изделию|размерная линия слишком далеко|"
                  "полка выноски не с той стороны|стрелка указывает не туда|"
                  "интервал шинки снят не с того участка|прочее")


class AdvisorError(RuntimeError):
    pass


def rules_path(cache: Path) -> Path:
    return cache / "правила.json"


def load_rules(cache: Path) -> list[dict]:
    path = rules_path(cache)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def save_rule(cache: Path, entry: dict) -> None:
    rules = load_rules(cache)
    rules = [r for r in rules if r.get("id") != entry.get("id")]
    rules.append(entry)
    path = rules_path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rules, ensure_ascii=False, indent=2), encoding="utf-8")


def hints(cache: Path, limit: int = 6, область: str = "рамка") -> list[str]:
    """Обобщённые правила из последних правок — подсказка для промпта уточнения.

    Правила разведены по области: подсказка про рамку декоративного элемента не
    должна попадать в разговор про раскладку линий и наоборот. Записи, сделанные
    до появления этого разделения, считаются правилами про рамку — тогда других
    и не было.
    """
    seen, out = set(), []
    for entry in reversed(load_rules(cache)):
        rule = (entry.get("правило") or "").strip()
        if entry.get("область", "рамка") != область:
            continue
        if entry.get("обобщается") and rule and rule not in seen:
            seen.add(rule)
            out.append(rule)
        if len(out) >= limit:
            break
    return out


def _encode(image: Image.Image) -> str:
    copy = image.copy()
    copy.thumbnail((VISION_SIDE, VISION_SIDE), Image.LANCZOS)
    buffer = io.BytesIO()
    copy.save(buffer, format="JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode()


MAX_TOKENS = 8000        # рассуждения Опуса съедают часть бюджета — держим запас


def call(content: list, api_key: str, system: str = SYSTEM, tools: list | None = None,
         messages: list | None = None, max_tokens: int = MAX_TOKENS,
         timeout: int = 600, effort: str | None = None) -> dict:
    """Один запрос к Опусу. Возвращает разобранный ответ целиком.

    `effort` — сколько модели думать перед ответом: low, medium, high, xhigh, max.
    Больше усилий даёт более выверенный разбор, но и ждать дольше.
    """
    import requests

    тело = {
        "model": ADVISOR_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": messages if messages is not None else [{"role": "user", "content": content}],
    }
    if tools:
        тело["tools"] = tools
    if effort:
        тело["output_config"] = {"effort": effort}
        тело["thinking"] = {"type": "adaptive"}
    ответ = requests.post(
        ANTHROPIC_BASE.rstrip("/") + "/messages", timeout=timeout,
        headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                 "Content-Type": "application/json"},
        json=тело)
    if ответ.status_code != 200:
        raise AdvisorError(f"HTTP {ответ.status_code}: {ответ.text[:300]}")
    return ответ.json()


def text_of(payload: dict) -> str:
    return "".join(b.get("text", "") for b in payload.get("content", [])
                   if b.get("type") == "text").strip()


def parse_json(payload: dict) -> dict:
    """Достаёт JSON из ответа. Обрыв по лимиту токенов называет своим именем."""
    текст = text_of(payload)
    начало, конец = текст.find("{"), текст.rfind("}")
    if начало >= 0 and конец > начало:
        try:
            return json.loads(текст[начало:конец + 1])
        except json.JSONDecodeError as exc:
            причина = f"JSON разобрать не удалось ({exc})"
    elif payload.get("stop_reason") == "max_tokens":
        причина = "ответ оборвался по лимиту токенов"
    elif not текст:
        причина = "модель вернула пустой ответ"
    else:
        причина = "в ответе нет JSON"
    хвост = " ".join(текст[-160:].split())
    raise AdvisorError(f"{причина}; конец ответа: …{хвост}" if хвост else причина)


def _ask(prompt: str, image: Image.Image, api_key: str, timeout: int = 600) -> dict:
    payload = call([
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": _encode(image)}},
        {"type": "text", "text": prompt},
    ], api_key, timeout=timeout)
    return parse_json(payload)


def _разница(авто, свой) -> list[str]:
    """Чем поправленный элемент разметки отличается от того, что дал автомат."""
    from .config import STYLE
    from .render import InclinedDim, Leader, LinearDim

    из: list[str] = []
    if type(авто) is not type(свой):
        как = "габаритом" if isinstance(свой, LinearDim) else "выноской"
        из.append(f"размер показан другим способом — {как}")
        return из
    if isinstance(свой, LinearDim):
        пролёт = max(abs(авто.b - авто.a), 1.0)
        for поле, имя in (("a", "начало"), ("b", "конец")):
            сдвиг = getattr(свой, поле) - getattr(авто, поле)
            if abs(сдвиг) > 3:
                из.append(f"{имя} габарита сдвинуто на {сдвиг:+.0f} px "
                          f"({сдвиг / пролёт * 100:+.0f} % длины)")
        if abs(свой.level - авто.level) > 3:
            шаг = abs(свой.level - авто.level) / STYLE.offset_1
            дальше = свой.side * (свой.level - авто.level) > 0
            куда = "дальше от изделия" if дальше else "ближе к изделию"
            из.append(f"размерная линия отодвинута {куда} на {шаг:.2f} эшелона")
        if свой.side != авто.side:
            из.append("размерная линия перенесена на другую сторону изделия")
    elif isinstance(свой, InclinedDim):
        for поле in ("a", "b"):
            сдвиг = math.dist(getattr(свой, поле), getattr(авто, поле))
            if сдвиг > 3:
                из.append(f"конец наклонного габарита сдвинут на {сдвиг:.0f} px")
        if abs(свой.offset - авто.offset) > 3:
            шаг = (свой.offset - авто.offset) / STYLE.offset_1
            из.append(f"размерная линия отодвинута от оси элемента на {шаг:+.2f} эшелона")
    elif isinstance(свой, Leader):
        сдвиг = math.dist(свой.elbow, авто.elbow)
        if сдвиг > 3:
            из.append(f"полка выноски переставлена на {сдвиг:.0f} px "
                      f"(в точку {свой.elbow[0]:.0f}, {свой.elbow[1]:.0f})")
        if свой.direction != авто.direction:
            из.append("полка развёрнута " + ("влево" if свой.direction < 0 else "вправо"))
        уход = math.dist(свой.anchor, авто.anchor)
        if уход > 3:
            из.append(f"стрелка переставлена на {уход:.0f} px "
                      f"(в точку {свой.anchor[0]:.0f}, {свой.anchor[1]:.0f})")
        if свой.span and авто.span and math.dist(_середина(свой.span),
                                                 _середина(авто.span)) > 3:
            из.append("интервал шинки снят с другого участка ободка")
        elif свой.span and not авто.span:
            из.append("размер показан интервалом на металле вместо стрелки")
    return из


def _середина(span) -> tuple:
    (x0, y0), (x1, y1) = span
    return ((x0 + x1) / 2, (y0 + y1) / 2)


def moves(space, item_id: str, plan) -> list[str]:
    """Правки раскладки словами: что человек подвинул и насколько.

    Сравнивается не «до и после мышки», а поправленный кадр с тем, что автомат
    дал бы сейчас: между правкой и разбором мог смениться и план, и код
    раскладки, а разбирать надо расхождение с сегодняшним автоматом.
    """
    from .layout import compose, find_decor
    from .planner import Plan

    if not plan.tweaks:
        return []
    row = space.rows[item_id]
    product = space.product(item_id)
    свободный = Plan.from_dict(plan.to_dict())
    свободный.tweaks = {}
    элемент, _ = find_decor(product, plan, row)
    эшелоны = getattr(space, "echelons", {}).get(item_id, {})
    # cache=None — сравниваем с чистым расчётом: привычки, выученные по прошлым
    # правкам, в разборе только запутали бы, они и есть след этих правок
    авто = compose(product, свободный, row, эшелоны, элемент, None)
    свой = compose(product, plan, row, эшелоны, элемент, None)
    описания = []
    for ключ in plan.tweaks:
        if ключ not in свой.keys or ключ not in авто.keys:
            continue
        строки = _разница(авто.items[авто.keys.index(ключ)],
                          свой.items[свой.keys.index(ключ)])
        if строки:
            описания.append(f"- «{CAPTIONS.get(ключ, ключ)}»: " + "; ".join(строки))
    return описания


def explain(space, item_id: str, body: dict) -> dict:
    """Просит Опуса объяснить, где автомат разошёлся с человеком.

    Разбираются обе породы правок: рамка декоративного элемента и перестановка
    самой разметки. Ответ ложится в `cache/правила.json` с пометкой области —
    подсказка про рамку не должна попадать в промпт про раскладку и наоборот.
    """
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        return {"ошибка": f"не задан ключ {API_KEY_ENV}"}

    row = space.rows[item_id]
    product = space.product(item_id)
    plan = space.plan(item_id)
    auto = body.get("auto") or {}
    manual = body.get("manual") or {}
    values = row.dims()

    def show(rect: dict) -> tuple:
        center = rect.get("center") or [rect.get("cx", 0), rect.get("cy", 0)]
        return ([round(float(v)) for v in center], round(float(rect.get("long", 0))),
                round(float(rect.get("short", 0))), round(float(rect.get("angle", 0))))

    auto_center, auto_long, auto_short, auto_angle = show(auto)
    man_center, man_long, man_short, man_angle = show(manual)

    сдвиг = max(abs(auto_center[0] - man_center[0]), abs(auto_center[1] - man_center[1]))
    размер = max(abs(auto_long - man_long), abs(auto_short - man_short))
    рамка_правлена = not (сдвиг < 8 and размер < 8 and abs(auto_angle - man_angle) < 3)

    переставлено = moves(space, item_id, plan)
    if not рамка_правлена and not переставлено:
        return {"ошибка": "автомат и правка совпадают — разбирать нечего"}

    части = [HEAD.format(item_type=plan.item_type, view=plan.view, canvas=product.size,
                         length=f"{values['length']:g}", width=f"{values['width']:g}",
                         height=f"{values['height']:g}", band_t=f"{values['band_t']:g}",
                         band_w=f"{values['band_w']:g}")]
    if рамка_правлена:
        части.append(RECT.format(
            auto_center=auto_center, auto_long=auto_long, auto_short=auto_short,
            auto_angle=auto_angle, auto_source=auto.get("source", "неизвестно"),
            manual_center=man_center, manual_long=man_long, manual_short=man_short,
            manual_angle=man_angle))
    if переставлено:
        части.append(MOVES.format(список="\n".join(переставлено)))
    # Область одна на разбор: правка рамки объясняет и перестановку линий, а
    # обратное неверно, поэтому при обеих правках правило пишется про рамку.
    область = "рамка" if рамка_правлена else "раскладка"
    части.append(TAIL.format(виды=ВИДЫ_РАМКИ if рамка_правлена else ВИДЫ_РАСКЛАДКИ))

    try:
        answer = _ask("\n\n".join(части), product.canvas, api_key)
    except (AdvisorError, ValueError, OSError) as exc:
        return {"ошибка": str(exc)}

    entry = {
        "id": item_id, "артикул": row.article, "тип": plan.item_type, "ракурс": plan.view,
        "когда": datetime.now().isoformat(timespec="seconds"),
        "область": область,
        "что_правили": sorted(plan.tweaks) if переставлено else [],
        "переставлено": переставлено,
        "авто": {"центр": auto_center, "длина": auto_long, "ширина": auto_short,
                 "наклон": auto_angle, "источник": auto.get("source", "")},
        "правка": {"центр": man_center, "длина": man_long, "ширина": man_short,
                   "наклон": man_angle},
        **{k: answer.get(k) for k in
           ("причина", "правило", "тип_ошибки", "масштаб_ошибки", "обобщается")},
    }
    save_rule(space.cache, entry)
    return {"разбор": entry, "правил_в_журнале": len(load_rules(space.cache))}
