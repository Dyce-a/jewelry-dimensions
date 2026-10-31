"""Планировщик разметки: что и куда ставить на конкретном фото.

Модель зрения не рисует изображение — она только отвечает на геометрические
вопросы (где декоративный элемент, где шинка, какой это ракурс). Все линии и
числа наносятся детерминированно, поэтому изделие остаётся неизменным, а
значения всегда совпадают с рабочей таблицей.
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from .config import (API_KEY_ENV, BAND_CUTS, DEFAULT_MODEL, HEIGHT_ON_LEADER,
                     PROXY_BASE, VISIBLE_KEYS)
from .imaging import Product

VISION_SIDE = 896          # до какого размера ужимаем картинку для модели

SYSTEM = (
    "Ты — инженер-технолог, который готовит ювелирные фото под нанесение "
    "размерных габаритов. Ты анализируешь геометрию снимка и отвечаешь строго "
    "одним JSON-объектом без пояснений."
)

PROMPT = """Это студийное фото ювелирного изделия на белом фоне (квадратный кадр).

Определи геометрию для нанесения размерных линий и верни JSON:

{{
  "item_type": "кольцо|серьги|серьга|подвеска|цепь|браслет|колье|брошь|пирсинг|прочее",
  "view": "фас|профиль|три четверти|сверху",
  "views_count": <сколько отдельных ракурсов/предметов видно в кадре, целое>,
  "decor_box": [x0, y0, x1, y1],
  "axes": {{"horizontal": "length|width|height", "vertical": "length|width|height"}},
  "band": {{
    "present": true|false,
    "point": [x, y],
    "side": "bottom|top|left|right"
  }},
  "band_width_visible": true|false,
  "problem": null | "коротко, почему габарит нельзя показать корректно"
}}

Правила:
- Координаты нормированные, 0..1, начало в левом верхнем углу кадра.
- "decor_box" — прямоугольник вокруг декоративной части: вставка с камнем, накладка,
  подвес, узор. Если изделие целиком декоративное (цепь, гладкое кольцо, серьга-кольцо),
  верни рамку всего изделия.
- "axes" сообщает, каким размерам из таблицы соответствуют ширина и высота decor_box
  НА ЭТОМ ракурсе. Третий размер не виден и будет показан выноской.
  Габариты изделия в миллиметрах: длина (length) = {length}, ширина (width) = {width},
  высота (height) = {height}. Соотнеси пропорции decor_box с этими числами.
- "band" — шинка кольца или дужка/замок серьги. "point" ставь туда, где её толщина
  видна как узкая перемычка; "side" — с какой стороны кадра эта точка.
  Если шинки нет, present=false и point=null.
- "band_width_visible" — видно ли ширину шинки как отдельный отрезок на этом ракурсе.
- "problem" заполняй только если ракурс действительно не позволяет показать габариты
  (изделие обрезано кадром, снято строго с торца, полностью перекрыто).

Отвечай только JSON."""

REFINE_PROMPT = """На кропе одно ювелирное изделие крупным планом.

Обведи декоративную часть целиком. Декоративная часть — это вставка вместе с кастом
(лапками, оправой, ободком из белого металла) и всей группой мелких камней вокруг
центрального: паве, ореол, дорожка на плечиках, если они образуют с центральным камнем
единый узел. Рамка обязана вместить весь этот узел от края до края.

Две ошибки, которых нужно избежать, и они одинаково грубые:
- рамка только по центральному камню, без каста и окружающих камней — это слишком мелко;
- рамка, заехавшая на шинку (гладкий ободок кольца), дужку или замок серьги — это слишком
  крупно. Шинка в рамку не входит даже там, где примыкает к вставке вплотную.

Проверь себя по масштабу. Декоративная часть — {decor} мм, толщина шинки — {band} мм.
Шинка на снимке работает линейкой: декоративная часть должна выйти примерно в {ratio}
раз(а) крупнее толщины шинки. Если рамка заметно мельче — в неё попал только камень,
расширь её до края каста; если заметно крупнее — в неё попал ободок, сузь.

Если декоративной части нет (гладкое кольцо, цепь, серьга-кольцо), обведи всё изделие
и поставь "whole_item": true.

Ответь JSON: {{"box": [x0, y0, x1, y1], "whole_item": true|false}}
в нормированных 0..1 координатах кропа."""


@dataclass
class Plan:
    """Разбор фото: где что лежит и какой размер каким способом показывать."""
    item_type: str = "прочее"
    view: str = "фас"
    views_count: int = 1
    decor_box: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)   # пиксели холста
    axes: dict = field(default_factory=lambda: {"horizontal": "width", "vertical": "length"})
    band_present: bool = True
    band_point: tuple[float, float] | None = None
    band_side: str = "bottom"
    band_width_visible: bool = False
    band_cuts: dict = field(default_factory=lambda: dict(BAND_CUTS))
    problem: str | None = None
    source: str = "geometry"        # 'vision' | 'geometry'
    decor_source: str = "rough"     # 'refined' — рамка уточнена вторым проходом по кропу
    notes: list = field(default_factory=list)
    raw: dict = field(default_factory=dict)
    canvas: int = 0                 # сторона холста, в пикселях которого записаны координаты
    manual: bool = False            # план поправлен человеком — модель его больше не трогает
    decor_lock: str = ""            # '', 'box', 'oriented': что считать декоративным элементом
    decor_rect: dict | None = None  # ориентированная рамка, если её задали вручную
    leader_anchor: list | None = None   # куда человек поставил стрелку высоты
    leader_lock: bool = False           # держать этот якорь, не пересчитывать
    proposed: str = ""                  # id образца, с которого перенесена правка
    #: Ручные поправки по каждому размеру: концы габарита, уровень размерной линии,
    #: полка и якорь выноски, интервал шинки, сторона полки. Ключ — размер из
    #: таблицы ('length', 'band_w', ...), значение — только те поля, которые человек
    #: действительно тронул. Всё остальное по-прежнему считает раскладка: правка
    #: одного конца не должна замораживать весь кадр.
    tweaks: dict = field(default_factory=dict)

    @property
    def leader_dim(self) -> str:
        """Размер декоративного элемента, который не виден на этом ракурсе."""
        used = {self.axes.get("horizontal"), self.axes.get("vertical")}
        for key in ("height", "width", "length"):
            if key not in used:
                return key
        return "height"

    # -- сохранение --------------------------------------------------
    #
    # План — это результат обращений к модели, самая дорогая часть прогона.
    # Он не зависит от того, как рисуются линии, поэтому кладётся в кэш:
    # правки отрисовки можно перепроверять сколько угодно раз бесплатно.

    def to_dict(self) -> dict:
        return {
            "item_type": self.item_type, "view": self.view,
            "views_count": self.views_count,
            "decor_box": list(self.decor_box), "axes": dict(self.axes),
            "band_present": self.band_present,
            "band_point": list(self.band_point) if self.band_point else None,
            "band_side": self.band_side,
            "band_width_visible": self.band_width_visible,
            "band_cuts": dict(self.band_cuts),
            "problem": self.problem, "source": self.source,
            "decor_source": self.decor_source,
            "notes": list(self.notes), "raw": self.raw,
            "canvas": self.canvas, "manual": self.manual,
            "decor_lock": self.decor_lock, "decor_rect": self.decor_rect,
            "leader_anchor": self.leader_anchor, "leader_lock": self.leader_lock,
            "proposed": self.proposed,
            "tweaks": dict(self.tweaks),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Plan":
        point = data.get("band_point")
        return cls(
            item_type=data.get("item_type", "прочее"),
            view=data.get("view", "фас"),
            views_count=int(data.get("views_count") or 1),
            decor_box=tuple(data.get("decor_box") or (0, 0, 1, 1)),
            axes=dict(data.get("axes") or {"horizontal": "width", "vertical": "length"}),
            band_present=bool(data.get("band_present", True)),
            band_point=tuple(point) if point else None,
            band_side=data.get("band_side", "bottom"),
            band_width_visible=bool(data.get("band_width_visible", False)),
            band_cuts={**BAND_CUTS, **(data.get("band_cuts") or {})},
            problem=data.get("problem"),
            source=data.get("source", "vision"),
            decor_source=data.get("decor_source", "rough"),
            notes=list(data.get("notes") or []),
            raw=dict(data.get("raw") or {}),
            canvas=int(data.get("canvas") or 0),
            manual=bool(data.get("manual", False)),
            decor_lock=str(data.get("decor_lock") or ""),
            decor_rect=data.get("decor_rect"),
            leader_anchor=data.get("leader_anchor"),
            leader_lock=bool(data.get("leader_lock", False)),
            proposed=str(data.get("proposed") or ""),
            tweaks=dict(data.get("tweaks") or {}),
        )

    def rescale(self, canvas: int) -> "Plan":
        """Пересчитывает координаты плана под другую сторону холста."""
        if not self.canvas or self.canvas == canvas:
            self.canvas = canvas
            return self
        k = canvas / self.canvas
        self.decor_box = tuple(v * k for v in self.decor_box)
        if self.band_point:
            self.band_point = (self.band_point[0] * k, self.band_point[1] * k)
        if self.leader_anchor:
            self.leader_anchor = [self.leader_anchor[0] * k, self.leader_anchor[1] * k]
        if self.decor_rect:
            rect = dict(self.decor_rect)
            rect["center"] = [rect["center"][0] * k, rect["center"][1] * k]
            rect["long"] = rect["long"] * k
            rect["short"] = rect["short"] * k
            self.decor_rect = rect
        if self.tweaks:
            self.tweaks = {key: rescale_tweak(поправка, k)
                           for key, поправка in self.tweaks.items()}
        self.canvas = canvas
        return self


#: Поля поправки, которые не являются координатами и пересчёту не подлежат.
NOT_PIXELS = ("direction",)


def rescale_tweak(поправка: dict, k: float) -> dict:
    """Пересчитывает ручную поправку под другую сторону холста.

    Всё, что в поправке записано числом, — это пиксели кадра, и при смене холста
    они обязаны поехать вместе с ним. Исключение одно: сторона полки — это знак,
    а не координата.
    """
    def сжать(значение):
        if isinstance(значение, (list, tuple)):
            return [сжать(v) for v in значение]
        if isinstance(значение, bool) or not isinstance(значение, (int, float)):
            return значение
        return значение * k

    return {поле: (значение if поле in NOT_PIXELS else сжать(значение))
            for поле, значение in поправка.items()}


#: Что человеку и Опусу позволено двигать на кадре. Больше в план не попадёт
#: ничего: координаты приходят из браузера, принимать их без разбора нельзя.
TWEAK_FIELDS = ("a", "b", "level", "offset", "anchor", "elbow", "direction", "span")


def _точка(значение, предел: float):
    if not isinstance(значение, (list, tuple)) or len(значение) != 2:
        return None
    try:
        x, y = float(значение[0]), float(значение[1])
    except (TypeError, ValueError):
        return None
    return [x, y] if abs(x) <= предел and abs(y) <= предел else None


def _число(значение, предел: float):
    try:
        число = float(значение)
    except (TypeError, ValueError):
        return None
    return число if abs(число) <= предел else None


def clean_tweak(поправка: dict, canvas: int) -> dict:
    """Проверяет одну правку размера: только известные поля и вменяемые координаты.

    Предел намеренно шире холста: полку выноски человек уводит за край кадра, и
    это законно. Отсекается не «за холстом», а «не координата вовсе».
    """
    предел = canvas * 4.0
    итог: dict = {}
    for поле in TWEAK_FIELDS:
        if поле not in поправка:
            continue
        значение = поправка[поле]
        if поле in ("level", "offset"):
            число = _число(значение, предел)
            if число is not None:
                итог[поле] = число
        elif поле in ("a", "b"):
            # у осевого габарита конец — одно число вдоль оси, у наклонного — точка
            найдено = _точка(значение, предел)
            найдено = найдено if найдено else _число(значение, предел)
            if найдено is not None:
                итог[поле] = найдено
        elif поле in ("anchor", "elbow"):
            точка = _точка(значение, предел)
            if точка:
                итог[поле] = точка
        elif поле == "direction":
            if значение in (-1, 1, "-1", "1"):
                итог[поле] = int(значение)
        elif поле == "span":
            if isinstance(значение, (list, tuple)) and len(значение) == 2:
                концы = [_точка(т, предел) for т in значение]
                if all(концы):
                    итог[поле] = концы
    return итог


# ---------------------------------------------------------------- fallback

def plan_geometric(product: Product, row) -> Plan:
    """Детерминированный план без обращения к модели.

    Декоративным элементом считается всё изделие; длина ложится на вертикаль,
    ширина — на горизонталь, высота и шинка уходят на выноски.
    """
    view = max(product.views, key=lambda v: v.area) if product.views else None
    x0, y0, x1, y1 = view.bbox if view else product.bbox
    tall = (y1 - y0) >= (x1 - x0)
    axes = ({"horizontal": "width", "vertical": "length"} if tall
            else {"horizontal": "length", "vertical": "width"})
    band_point = ((x0 + x1) / 2, y1 - (y1 - y0) * 0.04)
    return Plan(
        item_type="прочее",
        view="фас",
        views_count=len(product.views),
        decor_box=(x0, y0, x1, y1),
        axes=axes,
        band_present=row.band_t > 0 or row.band_w > 0,
        band_point=band_point,
        band_side="bottom",
        band_width_visible=False,
        source="geometry",
        canvas=product.size,
    )


# ---------------------------------------------------------------- vision

class VisionError(RuntimeError):
    pass


def _encode(image) -> str:
    img = (image.canvas if isinstance(image, Product) else image).copy()
    img.thumbnail((VISION_SIDE, VISION_SIDE), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode()


def _message(prompt: str, image, model: str, tokens: int = 6000) -> dict:
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": "data:image/jpeg;base64," + _encode(image)}},
            ]},
        ],
        "response_format": {"type": "json_object"},
        "max_completion_tokens": tokens,
    }


def _post(body: dict, api_key: str, timeout: int, retries: int = 3) -> dict:
    import requests

    url = PROXY_BASE.rstrip("/") + "/chat/completions"
    headers = {"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}
    last = None
    for attempt in range(retries):
        try:
            response = requests.post(url, headers=headers, json=body, timeout=timeout)
            if response.status_code in (429, 500, 502, 503, 504):
                raise VisionError(f"HTTP {response.status_code}: {response.text[:200]}")
            response.raise_for_status()
            return response.json()
        except Exception as exc:                      # noqa: BLE001
            last = exc
            if attempt < retries - 1:
                time.sleep(2 ** attempt * 1.5)
    raise VisionError(f"запрос к модели не удался: {last}")


def _clamp01(value, default=0.0) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _parse(data: dict, product: Product, row) -> Plan:
    size = product.size
    box = data.get("decor_box") or [0, 0, 1, 1]
    if len(box) != 4:
        box = [0, 0, 1, 1]
    x0, y0, x1, y1 = (_clamp01(v) * size for v in box)
    if x1 - x0 < size * 0.03 or y1 - y0 < size * 0.03:
        x0, y0, x1, y1 = product.bbox                  # рамка выродилась — берём всё изделие

    axes = data.get("axes") or {}
    horizontal = axes.get("horizontal") if axes.get("horizontal") in ("length", "width", "height") else None
    vertical = axes.get("vertical") if axes.get("vertical") in ("length", "width", "height") else None
    if not horizontal or not vertical or horizontal == vertical:
        tall = (y1 - y0) >= (x1 - x0)
        horizontal, vertical = ("width", "length") if tall else ("length", "width")

    band = data.get("band") or {}
    point = band.get("point")
    if isinstance(point, (list, tuple)) and len(point) == 2:
        band_point = (_clamp01(point[0]) * size, _clamp01(point[1]) * size)
    else:
        band_point = None

    problem = data.get("problem")
    if isinstance(problem, str) and not problem.strip():
        problem = None

    return Plan(
        item_type=str(data.get("item_type") or "прочее"),
        view=str(data.get("view") or "фас"),
        views_count=int(data.get("views_count") or len(product.views) or 1),
        decor_box=(x0, y0, x1, y1),
        axes={"horizontal": horizontal, "vertical": vertical},
        band_present=bool(band.get("present", row.band_t > 0 or row.band_w > 0)),
        band_point=band_point,
        band_side=str(band.get("side") or "bottom"),
        band_width_visible=bool(data.get("band_width_visible", False)),
        problem=problem,
        source="vision",
        raw=data,
        canvas=size,
    )


def _content(payload: dict) -> dict:
    """Разбирает ответ модели. Пустой content — исчерпанный бюджет рассуждений."""
    try:
        choice = payload["choices"][0]
        text = (choice["message"].get("content") or "").strip()
    except (KeyError, IndexError, TypeError) as exc:
        raise VisionError(f"неожиданный ответ модели: {exc}") from exc
    if not text:
        raise VisionError(
            f"модель вернула пустой ответ (finish_reason={choice.get('finish_reason')}) — "
            "рассуждающей модели не хватило max_completion_tokens")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise VisionError(f"модель вернула не JSON: {exc}") from exc


def _tighten(product: Product, box: tuple) -> tuple:
    """Поджимает рамку к фактическому металлу внутри неё."""
    import numpy as np

    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(product.size, x1), min(product.size, y1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return box
    sub = product.thin[y0:y1, x0:x1]
    if not sub.any():
        return box
    ys, xs = np.where(sub)
    return (x0 + int(xs.min()), y0 + int(ys.min()),
            x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)


DECOR_KEYS = ("length", "width", "height")
ASPECT_TOLERANCE = 0.30       # допустимое расхождение масштабов по двум осям, доли
CROP_SCALES = (0.85, 1.15)    # во сколько раз расширяем грубую рамку под уточнение
EXTRA_CROP_SCALE = 1.45       # третий замер, когда первые два разошлись


def aspect_fit(box: tuple, values: dict, prefer: dict | None = None) -> tuple[float, dict]:
    """Подбирает пару осей, при которой обе стороны рамки дают один масштаб мм/пиксель.

    Пропорции из рабочей таблицы здесь работают эталоном: если ширина рамки — это
    `a` мм, а высота — `b` мм, то a/ширина и b/высота обязаны совпасть. Возвращает
    (расхождение, оси); расхождение 0 — идеальное совпадение.
    """
    import math

    x0, y0, x1, y1 = box
    width_px, height_px = x1 - x0, y1 - y0
    if min(width_px, height_px) < 4:
        return float("inf"), prefer or {"horizontal": "width", "vertical": "length"}

    keys = VISIBLE_KEYS if HEIGHT_ON_LEADER else DECOR_KEYS
    best = (float("inf"), prefer or {"horizontal": "width", "vertical": "length"})
    for horizontal in keys:
        for vertical in keys:
            if horizontal == vertical:
                continue
            h_mm, v_mm = values.get(horizontal, 0.0), values.get(vertical, 0.0)
            if h_mm <= 0 or v_mm <= 0:
                continue
            score = abs(math.log((h_mm / width_px) / (v_mm / height_px)))
            axes = {"horizontal": horizontal, "vertical": vertical}
            if prefer and axes == prefer:
                score -= 1e-6           # при равенстве оставляем выбор модели
            if score < best[0]:
                best = (score, axes)
    return best


def _iou(a: tuple, b: tuple) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _refine_prompt(row, cache: "Path | None" = None) -> str:
    """Промпт уточнения с масштабной подсказкой: шинка служит линейкой в кадре.

    К нему добавляются правила, вытащенные из ручных правок: они описывают, на
    чём автомат уже промахивался, и работают как few-shot подсказки.
    """
    decor = "x".join(f"{v:g}" for v in (row.length, row.width) if v > 0) or f"{row.height:g}"
    band = max(row.band_t, 0.0) or 1.0
    ratio = max(row.length, row.width, row.height) / band
    prompt = REFINE_PROMPT.format(decor=decor, band=f"{band:g}", ratio=f"{ratio:.0f}")
    if cache is not None:
        try:
            from .advisor import hints
            rules = hints(cache)
        except Exception:                      # noqa: BLE001 — подсказки необязательны
            rules = []
        if rules:
            preface = "На таких же кадрах уже приходилось поправлять рамку вручную. Учти это:"
            prompt = "\n\n".join([prompt, preface, "\n".join(f"- {r}" for r in rules)])
    return prompt


def _ask_box(product: Product, origin: tuple, scale: float, prompt: str, model: str,
             api_key: str, timeout: int):
    """Один замер рамки по кропу вокруг `origin`. Возвращает (рамка, whole_item)."""
    x0, y0, x1, y1 = origin
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * scale
    crop_box = (max(0, cx - half), max(0, cy - half),
                min(product.size, cx + half), min(product.size, cy + half))
    if crop_box[2] - crop_box[0] < 32 or crop_box[3] - crop_box[1] < 32:
        return None, False

    crop = product.canvas.crop(tuple(int(v) for v in crop_box))
    data = _content(_post(_message(prompt, crop, model, tokens=4000), api_key, timeout))
    box = data.get("box")
    if not (isinstance(box, (list, tuple)) and len(box) == 4):
        return None, False

    w, h = crop.size
    candidate = _tighten(product, (
        crop_box[0] + _clamp01(box[0]) * w, crop_box[1] + _clamp01(box[1]) * h,
        crop_box[0] + _clamp01(box[2]) * w, crop_box[1] + _clamp01(box[3]) * h))
    floor = product.size * 0.015
    if candidate[2] - candidate[0] < floor or candidate[3] - candidate[1] < floor:
        return None, False
    return candidate, bool(data.get("whole_item"))


def refine_decor(product: Product, plan: Plan, row, model: str, api_key: str,
                 timeout: int = 180, cache: "Path | None" = None) -> None:
    """Уточняет рамку декоративного элемента двумя независимыми замерами по кропу.

    С полного кадра модель ставит рамку «на глаз» и регулярно прихватывает шинку —
    тогда габарит меряет не вставку. На кропе точность заметно выше, но ответ гуляет,
    поэтому замер делается дважды с разным полем зрения. Совпали — рамке можно верить;
    разошлись — берём более правдоподобную по пропорциям и помечаем строку к проверке.

    Сверять кандидата с рамкой первого прохода по пропорциям бесполезно: отношение
    сторон не зависит от масштаба, и завышенная рамка проходит такую проверку наравне
    с правильной.
    """
    import statistics

    prompt = _refine_prompt(row, cache)
    boxes, whole = [], False
    for scale in CROP_SCALES:
        candidate, is_whole = _ask_box(product, plan.decor_box, scale, prompt,
                                       model, api_key, timeout)
        whole = whole or is_whole
        if candidate:
            boxes.append(candidate)

    if whole:
        plan.notes.append("декоративный элемент отдельно не выделяется, размечен весь предмет")
    if not boxes:
        plan.notes.append("уточнить рамку не удалось, оставлена рамка первого прохода")
        return

    if len(boxes) == 2 and _iou(*boxes) >= 0.6:
        plan.decor_box = tuple(sum(v) / 2 for v in zip(*boxes))
    else:
        # замеры разошлись — добираем третий и берём покоординатную медиану:
        # она устойчива к одному выбросу, а именно так обычно и промахивается модель
        extra, _ = _ask_box(product, plan.decor_box, EXTRA_CROP_SCALE, prompt,
                            model, api_key, timeout)
        if extra:
            boxes.append(extra)
        if len(boxes) >= 3:
            plan.decor_box = tuple(statistics.median(v) for v in zip(*boxes))
        else:
            plan.decor_box = boxes[0]
            plan.notes.append("замеры рамки разошлись — стоит проверить границы вставки")
    plan.decor_box = _tighten(product, plan.decor_box)
    trimmed = product.dense_box(plan.decor_box)
    if _iou(trimmed, plan.decor_box) < 0.98:
        plan.decor_box = trimmed
    plan.decor_source = "refined"


def check_axes(plan: Plan, row) -> None:
    """Ставит оси по совпадению масштабов и отмечает случаи, где сойтись не удалось."""
    score, axes = aspect_fit(plan.decor_box, row.dims(), plan.axes)
    if HEIGHT_ON_LEADER and "height" in axes.values():
        # страховка на случай, если подбор не нашёл ни одной пары длина/ширина
        x0, y0, x1, y1 = plan.decor_box
        axes = ({"horizontal": "width", "vertical": "length"} if (y1 - y0) >= (x1 - x0)
                else {"horizontal": "length", "vertical": "width"})
    plan.axes = axes
    # Замечание о перекосе рамки ставит раскладка: она видит рамку уже после
    # поджима по маске и выправления по пропорциям, а здесь она ещё сырая.
    _ = score


def plan_with_vision(product: Product, row, model: str = DEFAULT_MODEL,
                     api_key: str | None = None, timeout: int = 180,
                     refine: bool = True, cache: "Path | None" = None) -> Plan:
    key = api_key or os.environ.get(API_KEY_ENV, "")
    if not key:
        raise VisionError(f"не задан ключ ProxyAPI (переменная {API_KEY_ENV})")

    prompt = PROMPT.format(length=row.length, width=row.width, height=row.height)
    plan = _parse(_content(_post(_message(prompt, product, model), key, timeout)), product, row)
    if refine and not plan.problem:
        try:
            refine_decor(product, plan, row, model, key, timeout, cache)
        except VisionError as exc:
            plan.notes.append(f"уточнение рамки не выполнено ({exc})")
    check_axes(plan, row)
    return plan


def build(product: Product, row, model: str = DEFAULT_MODEL, api_key: str | None = None,
          use_vision: bool = True, refine: bool = True,
          cache: "Path | None" = None) -> tuple[Plan, str | None]:
    """Возвращает (план, предупреждение). При отказе модели — геометрический план."""
    if not use_vision:
        return plan_geometric(product, row), None
    try:
        return plan_with_vision(product, row, model=model, api_key=api_key,
                                refine=refine, cache=cache), None
    except VisionError as exc:
        return plan_geometric(product, row), f"модель недоступна, разметка по геометрии ({exc})"
