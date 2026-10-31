"""Чат с Опусом, который сам правит разметку.

Опус получает те же ручки, что и человек в редакторе: посмотреть кадр, подвинуть
рамку декоративного элемента, переставить якорь выноски высоты, перерисовать,
найти похожие кадры и перенести на них правку. Он вызывает их сам, по одной, и
после каждого вызова видит новое состояние кадра — то есть работает так же, как
работал бы человек: посмотрел, подвинул, посмотрел ещё раз.

Границы намеренно жёсткие:

- инструментов ровно столько, сколько нужно для разметки; ни файлов, ни команд
  оболочки, ни доступа к рабочей таблице у него нет;
- массовая правка идёт только через `spread_fix`, а она помечает кадры как
  предложенные — принимает их всё равно человек;
- любую правку снимает `reset_frame`, поэтому испортить исходные данные нельзя:
  фотографии не трогаются, меняется только план разметки.

Так «мелочи» вроде «подвинь рамку правее и перерисуй» решаются в чате, а не
превращаются в отдельную задачу.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .advisor import AdvisorError, _encode, call
from .config import API_KEY_ENV

MAX_STEPS = 12          # предохранитель от зацикливания
HISTORY_LIMIT = 40      # сколько сообщений держим в переписке

SYSTEM = """Ты — инженер-технолог, который правит разметку размерных габаритов на
фотографиях ювелирных изделий вместе с оператором.

Как устроена разметка. Числа берутся из рабочей таблицы и не обсуждаются. Линии
рисует код по «плану» кадра. Главное в плане — рамка декоративного элемента: по
ней меряются длина и ширина, от неё же отталкивается выноска высоты. Рамка
задаётся центром, длинной и короткой стороной и наклоном в градусах; координаты —
пиксели квадратного холста, начало в левом верхнем углу, ось Y вниз.

Что считается декоративным элементом: вставка вместе с кастом, лапками и группой
мелких камней, образующих с ней единый узел. Шинка (гладкий ободок кольца),
дужка и замок серьги в рамку не входят.

Кроме рамки правится и сама раскладка. Любой из пяти размеров можно переставить
`set_dimension`: развести концы габарита, отодвинуть размерную линию от изделия,
перенести полку выноски или развернуть её в другую сторону. Числа при этом не
меняются никогда — они из таблицы; правка говорит только о том, где линия стоит.
Передавай в `set_dimension` лишь те поля, которые действительно двигаешь.

Как работать. Сначала посмотри кадр инструментом `inspect_frame` — там и снимок, и
все числа. Правь по одному шагу и после каждого смотри результат. Если оператор
просит поправить «на глаз», опирайся на снимок и на сверку масштаба: мм/пиксель по
длинной и короткой стороне обязаны совпасть, расхождение больше 18 % — признак
неверной рамки.

Кроме разметки ты правишь сам этот редактор — его разметку, стили и скрипт в
каталоге `web/`. Перед правкой обязательно прочитай файл: заменять кусок вслепую
нельзя. Правь точечно инструментом `patch_code`, целиком файл переписывай только
когда иначе никак. После записи файл проверяется на синтаксис, и при поломке
правка откатывается сама — но лучше до этого не доводить.

Отвечай по-русски, коротко и по делу: что сделал и что получилось. Не пересказывай
содержимое инструментов целиком. На вопросы отвечай прямо, без предисловий."""

TOOLS = [
    {
        "name": "inspect_frame",
        "description": ("Снимок изделия и полный разбор: рамка декоративного элемента, "
                        "масштаб, промер шинки, замечания раскладки, размеры из таблицы."),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string", "description": "ID позиции, например S2-131"}},
            "required": ["id"],
        },
    },
    {
        "name": "set_decor_rect",
        "description": ("Задать рамку декоративного элемента и перерисовать кадр. Рамка "
                        "закрепляется за человеком: автоматика её больше не трогает. "
                        "cx, cy — центр в пикселях холста; long и short — стороны; "
                        "angle — наклон длинной стороны в градусах, 0 вдоль горизонтали."),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "cx": {"type": "number"},
                "cy": {"type": "number"},
                "long": {"type": "number"},
                "short": {"type": "number"},
                "angle": {"type": "number"},
            },
            "required": ["id", "cx", "cy", "long", "short"],
        },
    },
    {
        "name": "set_height_anchor",
        "description": ("Куда смотрит стрелка выноски «Высота декоративного элемента». "
                        "Точка притягивается к ближайшему металлу внутри рамки."),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "x": {"type": "number"},
                           "y": {"type": "number"}},
            "required": ["id", "x", "y"],
        },
    },
    {
        "name": "set_dimension",
        "description": ("Переставить конкретный размер на кадре: концы габарита, уровень "
                        "размерной линии, полку и стрелку выноски, сторону полки. "
                        "Передавай только те поля, которые действительно двигаешь, — "
                        "остальное досчитает раскладка. Координаты в пикселях холста."),
        "input_schema": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "размер": {"type": "string",
                           "description": "length, width, height, band_w или band_t"},
                "a": {"type": "number", "description": "начало габарита вдоль его оси"},
                "b": {"type": "number", "description": "конец габарита вдоль его оси"},
                "level": {"type": "number",
                          "description": "координата размерной линии поперёк её оси"},
                "elbow": {"type": "array", "items": {"type": "number"},
                          "description": "излом выноски: [x, y]"},
                "anchor": {"type": "array", "items": {"type": "number"},
                           "description": "куда смотрит стрелка выноски: [x, y]"},
                "direction": {"type": "integer",
                              "description": "сторона полки: -1 влево, 1 вправо"},
            },
            "required": ["id", "размер"],
        },
    },
    {
        "name": "reset_dimension",
        "description": ("Вернуть один размер под автоматику, не трогая остальные правки кадра."),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"}, "размер": {"type": "string"}},
            "required": ["id", "размер"],
        },
    },
    {
        "name": "rerender",
        "description": "Перерисовать кадр по текущему плану, ничего не меняя.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "string"}},
                         "required": ["id"]},
    },
    {
        "name": "reset_frame",
        "description": "Снять все ручные правки с кадра и вернуть автоматическую разметку.",
        "input_schema": {"type": "object", "properties": {"id": {"type": "string"}},
                         "required": ["id"]},
    },
    {
        "name": "similar_frames",
        "description": ("Кадры, где автомат мог ошибиться так же: тот же тип изделия, "
                        "тот же ракурс, близкие табличные пропорции."),
        "input_schema": {
            "type": "object",
            "properties": {"id": {"type": "string"},
                           "limit": {"type": "integer", "description": "по умолчанию 12"}},
            "required": ["id"],
        },
    },
    {
        "name": "spread_fix",
        "description": ("Перенести правку с образца на перечисленные кадры. Кадры "
                        "перерисовываются и помечаются как предложенные: принимает их человек."),
        "input_schema": {
            "type": "object",
            "properties": {
                "source": {"type": "string", "description": "ID образца с ручной правкой"},
                "ids": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["source", "ids"],
        },
    },
]

CODE_TOOLS = [
    {
        "name": "list_code",
        "description": "Файлы проекта: путь, размер, число строк. По умолчанию каталог web.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string", "description": "каталог или файл, например web"}},
        },
    },
    {
        "name": "read_code",
        "description": ("Прочитать файл с номерами строк. Можно взять кусок: from_line "
                        "и lines. Перед любой правкой читай файл."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "from_line": {"type": "integer"},
                "lines": {"type": "integer"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "patch_code",
        "description": ("Заменить кусок текста в файле. old обязан встречаться ровно "
                        "один раз — берите достаточно контекста. После записи файл "
                        "проверяется, при поломке правка откатывается."),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string"},
                "new": {"type": "string"},
                "message": {"type": "string", "description": "коротко, что за правка"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "write_code",
        "description": "Переписать файл целиком. Только когда точечной правкой не обойтись.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "code_history",
        "description": "Последние правки кода и кто их сделал.",
        "input_schema": {"type": "object",
                         "properties": {"limit": {"type": "integer"}}},
    },
    {
        "name": "undo_code",
        "description": ("Снять последние собственные правки кода. Чужие коммиты не "
                        "трогает. Пользуйся, если правка вышла неудачной."),
        "input_schema": {"type": "object",
                         "properties": {"count": {"type": "integer"}}},
    },
]

#: Русские подписи для журнала в интерфейсе.
TOOL_LABELS = {
    "inspect_frame": "посмотрел кадр",
    "set_decor_rect": "подвинул рамку",
    "set_height_anchor": "переставил якорь высоты",
    "set_dimension": "переставил размер",
    "reset_dimension": "вернул размер автомату",
    "rerender": "перерисовал",
    "reset_frame": "вернул автоматику",
    "similar_frames": "нашёл похожие",
    "spread_fix": "перенёс правку",
    "list_code": "посмотрел файлы",
    "read_code": "прочитал код",
    "patch_code": "поправил код",
    "write_code": "переписал файл",
    "code_history": "посмотрел историю правок",
    "undo_code": "откатил правку",
}


class ToolError(RuntimeError):
    pass


# ---------------------------------------------------------------- инструменты

def _разбор(space, item_id: str) -> dict:
    from .inspect import describe

    row = space.rows[item_id]
    product = space.product(item_id)
    plan = space.plan(item_id)
    данные = describe(row, plan, product, space.cache)
    return {
        "id": item_id, "номенклатура": row.article,
        "размеры_мм": row.dims(),
        "холст": данные["canvas"],
        "рамка": {"центр": данные["decor"]["center"],
                  "длинная_сторона": данные["decor"]["long"],
                  "короткая_сторона": данные["decor"]["short"],
                  "наклон": данные["decor"]["angle"],
                  "источник": данные["decor"]["source"],
                  "наклонные_габариты": данные["decor"]["inclined"]},
        "якорь_высоты": данные["leader"],
        "масштаб": данные["scale"],
        "шинка": данные["band"],
        "тип": plan.item_type, "ракурс": plan.view,
        # что уже нарисовано на кадре: без этих координат `set_dimension` работал
        # бы вслепую — двигать линию, не зная, где она стоит, нельзя
        "разметка": [_фигура(ф) for ф in данные["shapes"]],
        "замечания": данные["notes"],
    }


def _фигура(ф: dict) -> dict:
    """Одна линия разметки в том виде, в каком её можно двигать инструментом."""
    итог = {"размер": ф["key"], "число": ф["text"], "правлен": ф["tweaked"]}
    if ф["kind"] == "linear":
        итог.update({"как": "габарит", "ось": ф["axis"],
                     "концы": [round(ф["a"], 1), round(ф["b"], 1)],
                     "уровень": round(ф["level"], 1)})
    elif ф["kind"] == "inclined":
        итог.update({"как": "наклонный габарит", "концы": [ф["a"], ф["b"]],
                     "вынос": round(ф["offset"], 1)})
    else:
        итог.update({"как": "интервал" if ф["span"] else "выноска",
                     "полка": [round(v, 1) for v in ф["elbow"]],
                     "стрелка": [round(v, 1) for v in ф["anchor"]],
                     "сторона": ф["direction"]})
    return итог


def _кадр_с_картинкой(space, item_id: str) -> list:
    разбор = _разбор(space, item_id)
    product = space.product(item_id)
    return [
        {"type": "text", "text": json.dumps(разбор, ensure_ascii=False)},
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                     "data": _encode(product.canvas)}},
    ]


def _проверить_id(space, item_id: str) -> str:
    if item_id not in space.rows:
        raise ToolError(f"позиции {item_id} нет в рабочей таблице")
    return item_id


def _запомнить_привычку(space, item_id: str, план, ключи) -> None:
    """Кладёт правку размера в журнал привычек — тот же путь, что и у правки мышью."""
    from . import habits
    from .layout import compose, find_decor

    try:
        row = space.rows[item_id]
        product = space.product(item_id)
        элемент, _ = find_decor(product, план, row)
        лист = compose(product, план, row, getattr(space, "echelons", {}).get(item_id, {}),
                       элемент, space.cache)
        habits.remember(space.cache, item_id, план, product, лист, set(ключи))
    except Exception:                          # noqa: BLE001 — журнал необязателен
        pass


def выполнить(space, имя: str, аргументы: dict) -> list:
    """Выполняет один инструмент и возвращает содержимое ответа для модели."""
    if имя == "inspect_frame":
        return _кадр_с_картинкой(space, _проверить_id(space, аргументы["id"]))

    if имя == "set_decor_rect":
        item_id = _проверить_id(space, аргументы["id"])
        план = space.plan(item_id)
        план.decor_rect = {
            "center": [float(аргументы["cx"]), float(аргументы["cy"])],
            "long": float(аргументы["long"]),
            "short": float(аргументы["short"]),
            "angle": float(аргументы.get("angle") or 0.0),
        }
        план.decor_lock = "oriented"
        план.manual = True
        половина = max(план.decor_rect["long"], план.decor_rect["short"]) / 2
        cx, cy = план.decor_rect["center"]
        план.decor_box = (cx - половина, cy - половина, cx + половина, cy + половина)
        space.save_plan(item_id, план)
        space.render(item_id)
        return _кадр_с_картинкой(space, item_id)

    if имя == "set_height_anchor":
        item_id = _проверить_id(space, аргументы["id"])
        from . import anchors
        from .layout import find_decor

        план = space.plan(item_id)
        точка = [float(аргументы["x"]), float(аргументы["y"])]
        план.leader_anchor = точка
        план.leader_lock = True
        план.manual = True
        элемент, _ = find_decor(space.product(item_id), план, space.rows[item_id])
        anchors.remember(space.cache, item_id, план.item_type, план.view, точка, элемент)
        space.save_plan(item_id, план)
        space.render(item_id)
        return _кадр_с_картинкой(space, item_id)

    if имя in ("set_dimension", "reset_dimension"):
        from . import habits
        from .config import DIMENSIONS
        from .planner import clean_tweak

        item_id = _проверить_id(space, аргументы["id"])
        размер = str(аргументы.get("размер") or "")
        if размер not in dict(DIMENSIONS):
            raise ToolError(f"нет такого размера: {размер}. "
                            f"Годятся: {', '.join(k for k, _ in DIMENSIONS)}")
        план = space.plan(item_id)
        правки = dict(план.tweaks or {})
        if имя == "reset_dimension":
            правки.pop(размер, None)
            habits.forget(space.cache, item_id, {размер})
        else:
            поправка = clean_tweak(аргументы, space.canvas)
            if not поправка:
                raise ToolError("не передано ни одного поля правки: "
                                "a, b, level, elbow, anchor или direction")
            правки[размер] = {**правки.get(размер, {}), **поправка}
        план.tweaks = правки
        план.manual = True
        space.save_plan(item_id, план)
        space.render(item_id)
        # привычка записывается тем же порядком, что и после правки мышью:
        # для журнала неважно, чья рука подвинула линию
        if имя == "set_dimension":
            _запомнить_привычку(space, item_id, план, {размер})
        return _кадр_с_картинкой(space, item_id)

    if имя == "rerender":
        item_id = _проверить_id(space, аргументы["id"])
        space.render(item_id)
        return _кадр_с_картинкой(space, item_id)

    if имя == "reset_frame":
        item_id = _проверить_id(space, аргументы["id"])
        план = space.plan(item_id)
        план.manual = False
        план.proposed = ""
        план.decor_lock, план.decor_rect = "", None
        план.leader_lock, план.leader_anchor = False, None
        план.tweaks = {}
        from . import habits
        habits.forget(space.cache, item_id)
        space.save_plan(item_id, план)
        space.render(item_id)
        return _кадр_с_картинкой(space, item_id)

    if имя == "similar_frames":
        from . import spread

        item_id = _проверить_id(space, аргументы["id"])
        сколько = int(аргументы.get("limit") or spread.DEFAULT_LIMIT)
        найдено = spread.candidates(space, item_id, сколько)
        return [{"type": "text", "text": json.dumps(
            {"похожие": [{"id": i, "номенклатура": space.rows[i].article,
                          "размеры_мм": space.rows[i].dims()} for i in найдено]},
            ensure_ascii=False)}]

    if имя == "spread_fix":
        from . import spread
        from .layout import find_decor
        from .planner import Plan

        образец = _проверить_id(space, аргументы["source"])
        план = space.plan(образец)
        if not (план.manual and (план.decor_rect or план.tweaks)):
            raise ToolError("у образца нет ручной правки — переносить нечего")
        поправка = {}
        if план.decor_rect:
            свободный = Plan.from_dict(план.to_dict())
            свободный.decor_lock, свободный.decor_rect, свободный.manual = "", None, False
            авто, _ = find_decor(space.product(образец), свободный, space.rows[образец])
            поправка = spread.transform(
                {"center": авто.center, "long": авто.long, "short": авто.short,
                 "angle": авто.angle}, план.decor_rect)
        правки = spread.tweaks_of(space, образец)
        if правки:
            поправка["поправки"] = правки
        кадры = [_проверить_id(space, i) for i in аргументы.get("ids") or []]
        сделано = spread.propose(space, кадры, поправка, образец)
        return [{"type": "text", "text": json.dumps(
            {"поправка": поправка, "результат": сделано,
             "примечание": "кадры помечены как предложенные, принимает их человек"},
            ensure_ascii=False)}]

    if имя in ("list_code", "read_code", "patch_code", "write_code",
               "code_history", "undo_code"):
        return _код(space, имя, аргументы)

    raise ToolError(f"неизвестный инструмент {имя}")


def _код(space, имя: str, аргументы: dict) -> list:
    """Инструменты правки исходников редактора."""
    from . import devtools

    гараж = getattr(space, "гараж", None) or devtools.Гараж()

    def ответ(данные) -> list:
        return [{"type": "text", "text": json.dumps(данные, ensure_ascii=False)}]

    try:
        if имя == "list_code":
            return ответ({"файлы": devtools.перечислить(гараж, аргументы.get("path") or "web"),
                          "писать_можно_в": list(гараж.корни)})
        if имя == "read_code":
            return ответ(devtools.прочитать(гараж, аргументы["path"],
                                            int(аргументы.get("from_line") or 1),
                                            int(аргументы.get("lines") or 0)))
        if имя == "patch_code":
            return ответ(devtools.заменить(гараж, аргументы["path"], аргументы["old"],
                                           аргументы["new"], аргументы.get("message", "")))
        if имя == "write_code":
            return ответ(devtools.записать(гараж, аргументы["path"], аргументы["content"],
                                           аргументы.get("message", "")))
        if имя == "code_history":
            return ответ({"история": devtools.история(int(аргументы.get("limit") or 10))})
        if имя == "undo_code":
            return ответ({"итог": devtools.откатить(int(аргументы.get("count") or 1))})
    except devtools.DevError as exc:
        raise ToolError(str(exc)) from exc
    raise ToolError(f"неизвестный инструмент {имя}")


# ---------------------------------------------------------------- цикл беседы

def _очистить(история: list) -> list:
    """Держим переписку в разумных пределах, не разрывая пары «вызов — ответ»."""
    if len(история) <= HISTORY_LIMIT:
        return история
    обрезок = история[-HISTORY_LIMIT:]
    while обрезок and обрезок[0].get("role") != "user":
        обрезок = обрезок[1:]
    return обрезок


EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")


def chat(space, история: list, item_id: str | None = None,
         effort: str = "medium") -> dict:
    """Проводит один ход беседы: модель может несколько раз вызвать инструменты."""
    api_key = os.environ.get(API_KEY_ENV, "")
    if not api_key:
        return {"ошибка": f"не задан ключ {API_KEY_ENV}"}

    гараж = getattr(space, "гараж", None)
    system = SYSTEM
    if item_id:
        system += (f"\n\nСейчас оператор смотрит позицию {item_id}. "
                   "Если он не называет другой ID, работай с ней.")
    if гараж and гараж.можно_писать:
        system += ("\n\nПисать в исходники можно только в: "
                   + ", ".join(гараж.корни) + ".")
    else:
        system += ("\n\nПравка кода сейчас выключена: инструментами `patch_code` и "
                   "`write_code` пользоваться нельзя, объясни это оператору.")

    инструменты = list(TOOLS) + (CODE_TOOLS if гараж and гараж.можно_писать
                                 else CODE_TOOLS[:2] + CODE_TOOLS[4:])
    if effort not in EFFORT_LEVELS:
        effort = "medium"

    сообщения = _очистить(list(история))
    журнал: list[dict] = []

    for шаг in range(MAX_STEPS):
        try:
            ответ = call([], api_key, system=system, tools=инструменты,
                         messages=сообщения, effort=effort)
        except AdvisorError as exc:
            return {"ошибка": str(exc), "история": сообщения, "журнал": журнал}

        содержимое = ответ.get("content") or []
        сообщения.append({"role": "assistant", "content": содержимое})

        вызовы = [b for b in содержимое if b.get("type") == "tool_use"]
        if not вызовы:
            текст = "".join(b.get("text", "") for b in содержимое if b.get("type") == "text")
            return {"ответ": текст.strip(), "история": сообщения, "журнал": журнал,
                    "усилие": effort}

        результаты = []
        for вызов in вызовы:
            имя, аргументы = вызов.get("name", ""), вызов.get("input") or {}
            запись = {"инструмент": TOOL_LABELS.get(имя, имя), "аргументы": аргументы}
            try:
                содержимое_ответа = выполнить(space, имя, аргументы)
                запись["итог"] = "выполнено"
            except Exception as exc:                  # noqa: BLE001 — ошибку показываем модели
                содержимое_ответа = [{"type": "text", "text": f"ошибка: {exc}"}]
                запись["итог"] = f"ошибка: {exc}"
            журнал.append(запись)
            результаты.append({"type": "tool_result", "tool_use_id": вызов.get("id"),
                               "content": содержимое_ответа})
        сообщения.append({"role": "user", "content": результаты})

    return {"ответ": "Слишком много шагов подряд — остановился, посмотрите результат.",
            "история": сообщения, "журнал": журнал}
