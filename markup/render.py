"""Отрисовка размерных выносок в стиле эталонов SOKOLOV."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from .config import CANVAS, FONT_DIR, STYLE, UNIT
from .imaging import Product

TEXT_PAD = 4          # поле вокруг глифов в слое текста, px
EXT_RUNAWAY = 4       # во сколько раз выносной позволено превысить засечку, прежде чем счесть промах

FONT_CANDIDATES = {
    "label": ["Montserrat-SemiBold.ttf", "Montserrat[wght].ttf",
              "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf"],
    "caption": ["Montserrat-Medium.ttf", "Montserrat[wght].ttf",
                "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf"],
}


@lru_cache(maxsize=32)
def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for name in FONT_CANDIDATES[kind]:
        path = Path(name) if name.startswith("C:/") else FONT_DIR / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def fmt(value: float) -> str:
    """«12 мм», «4,5 мм» — как в исходной таблице, с русской запятой."""
    text = f"{value:g}".replace(".", ",")
    return text + " " + UNIT


def measure(text: str, f: ImageFont.FreeTypeFont) -> tuple[int, int]:
    box = f.getbbox(text)
    return box[2] - box[0], box[3] - box[1]


def _upright(ux: float, uy: float) -> int:
    """Угол поворота подписи вдоль направления (ux, uy), головой вверх.

    Экранная ось Y смотрит вниз, поэтому поворот берётся со знаком минус;
    перевёрнутые вниз головой варианты доворачиваются на 180°.
    """
    degrees = -math.degrees(math.atan2(uy, ux))
    while degrees <= -90:
        degrees += 180
    while degrees > 90:
        degrees -= 180
    return round(degrees)


def _text_layer(text: str, f: ImageFont.FreeTypeFont, color, rotate: int = 0) -> Image.Image:
    box = f.getbbox(text)
    pad = TEXT_PAD
    layer = Image.new("RGBA", (box[2] - box[0] + 2 * pad, box[3] - box[1] + 2 * pad), (0, 0, 0, 0))
    ImageDraw.Draw(layer).text((pad - box[0], pad - box[1]), text, font=f, fill=color)
    if rotate:
        layer = layer.rotate(rotate, expand=True, resample=Image.BICUBIC)
    return layer


# ---------------------------------------------------------------- элементы разметки

@dataclass
class LinearDim:
    """Габаритный размер: две выносные линии и размерная линия с числом."""
    axis: str          # 'h' — меряем вдоль X, 'v' — вдоль Y
    a: float           # начало измеряемого отрезка
    b: float           # конец
    base: float        # запасная кромка изделия, если её не удалось нащупать лучом
    level: float       # координата размерной линии
    text: str
    side: int          # -1: размерная линия сверху/слева от изделия, +1: снизу/справа
    snap: bool = True  # искать кромку под каждым концом отдельно
    box: tuple | None = None   # рамка измеряемого элемента: в ней и ищется кромка


@dataclass
class InclinedDim:
    """Габарит вдоль собственной оси декоративного элемента.

    Вытянутый декор, лежащий по диагонали, описанным прямоугольником не меряется:
    прямоугольник заметно длиннее самого элемента. Поэтому размерная линия идёт
    параллельно оси элемента, а выносные — по нормали к ней.
    """
    a: tuple           # начало измеряемого отрезка, на кромке элемента
    b: tuple           # конец
    normal: tuple      # единичная нормаль: куда выносится размерная линия
    offset: float      # на сколько размерная линия отодвинута от оси элемента
    text: str
    overshoot: float = 0.0   # насколько продлить луч поиска кромки за ось элемента


@dataclass
class Leader:
    """Выноска-полка: размер, который не показать габаритом на этом ракурсе."""
    anchor: tuple[float, float]    # точка на изделии
    elbow: tuple[float, float]     # излом, начало полки
    direction: int                 # -1 полка влево, +1 вправо
    length: float
    text: str
    caption: str = ""
    arrow: str = ""                # '', 'v', 'h' — стрелка у якоря
    span: tuple | None = None      # интервал поперёк детали: две точки на её кромках
    hit_box: tuple | None = None   # где засчитывать касание контура (рамка вставки)
    overshoot: float = 0.0         # насколько продлить луч за якорь в поисках металла
    snap: bool = True              # досаживать конец лучом; False — якорь уже на контуре


def _flip(dim: LinearDim) -> int:
    """С какой стороны от измеряемого элемента оказалась размерная линия.

    Сторона задаёт, куда пойдут выносные линии, и пересчитывается всякий раз,
    когда линию двигают руками: перетащенная на другую сторону изделия, она иначе
    стреляла бы выносными в противоположную сторону — мимо элемента.
    """
    if not dim.box:
        return dim.side
    середина = ((dim.box[1] + dim.box[3]) / 2 if dim.axis == "h"
                else (dim.box[0] + dim.box[2]) / 2)
    return -1 if dim.level < середина else +1


def подправить(item, поправка: dict):
    """Накладывает ручную поправку человека на элемент разметки.

    Поправка описывает только то, что человек тронул мышью: конец габарита,
    уровень размерной линии, полку или якорь выноски, интервал шинки, сторону
    полки. Остальные поля элемента остаются как их посчитала раскладка — иначе
    правка одного конца замораживала бы весь кадр целиком.
    """
    if not поправка:
        return item
    if isinstance(item, LinearDim):
        for поле in ("a", "b", "level"):
            if isinstance(поправка.get(поле), (int, float)):
                setattr(item, поле, float(поправка[поле]))
        if "level" in поправка:
            item.side = _flip(item)
    elif isinstance(item, InclinedDim):
        for поле in ("a", "b"):
            точка = поправка.get(поле)
            if isinstance(точка, (list, tuple)) and len(точка) == 2:
                setattr(item, поле, (float(точка[0]), float(точка[1])))
        if isinstance(поправка.get("offset"), (int, float)):
            item.offset = float(поправка["offset"])
    elif isinstance(item, Leader):
        for поле in ("anchor", "elbow"):
            точка = поправка.get(поле)
            if isinstance(точка, (list, tuple)) and len(точка) == 2:
                setattr(item, поле, (float(точка[0]), float(точка[1])))
        if "anchor" in поправка:
            # человек показал пальцем на конкретный пиксель; досаживать этот конец
            # лучом от полки значит увести его на первый попавшийся металл
            item.snap = False
        if поправка.get("direction") in (-1, 1):
            item.direction = int(поправка["direction"])
        интервал = поправка.get("span")
        if isinstance(интервал, (list, tuple)) and len(интервал) == 2:
            item.span = tuple((float(x), float(y)) for x, y in интервал)
            item.snap = False
            if "anchor" not in поправка:
                # полка ведётся от ближнего к излому конца интервала: переставленный
                # интервал иначе остался бы связан с полкой по старому месту
                середина = ((item.span[0][0] + item.span[1][0]) / 2,
                            (item.span[0][1] + item.span[1][1]) / 2)
                item.anchor = min((item.span[0], item.span[1], середина),
                                  key=lambda точка: math.dist(точка, item.elbow))
    return item


@dataclass
class Sheet:
    """Итоговый лист: изделие плюс список элементов разметки."""
    product: Product
    items: list = field(default_factory=list)
    keys: list = field(default_factory=list)       # какой размер отвечает за каждый элемент
    notes: list = field(default_factory=list)      # замечания раскладки
    anchor_source: str = ""                        # откуда взят якорь выноски высоты
    #: Ручные поправки по размерам: {'length': {'a': ..., 'level': ...}, ...}.
    #: Накладываются здесь, в одном месте, а не в каждой ветке раскладки: размер
    #: может уйти с габарита на выноску и обратно, и поправка обязана ехать за ним.
    tweaks: dict = field(default_factory=dict)
    #: Какие размеры действительно поправлены — редактору, чтобы показать это глазами.
    tweaked: list = field(default_factory=list)

    def add(self, item, key: str = "") -> None:
        """Кладёт элемент разметки, помечая его размером из таблицы.

        Метка нужна разбору конфликтов: раньше размер восстанавливался по номеру
        элемента в списке, а порядок элементов постоянным быть перестал — размер,
        не сошедшийся с кадром, уходит с габарита на выноску и меняет очередь.
        Пересчитывать по номеру теперь значит винить в наложении подписей не тот
        размер и раздвигать не тот эшелон.
        """
        поправка = self.tweaks.get(key)
        if поправка:
            item = подправить(item, поправка)
            if key not in self.tweaked:
                self.tweaked.append(key)
        self.items.append(item)
        self.keys.append(key)

    def key_at(self, index: int) -> str:
        return self.keys[index] if 0 <= index < len(self.keys) else ""


# ---------------------------------------------------------------- рисование

class Renderer:
    def __init__(self, style=STYLE, size: int = CANVAS):
        self.st = style
        self.size = size
        self.label_font = font("label", round(style.label_size))
        self.caption_font = font("caption", round(style.caption_size))
        self.segments: list = []      # отрезки текущего листа, для контроля пересечений

    # -- низкоуровневые примитивы --------------------------------------

    def _line(self, d: ImageDraw.ImageDraw, p1, p2, color=None, width=None) -> None:
        d.line([p1, p2], fill=color or self.st.line_color, width=width or self.st.line_width)
        self.segments.append((p1[0], p1[1], p2[0], p2[1]))

    def _dot(self, d: ImageDraw.ImageDraw, p) -> None:
        r = self.st.dot_radius
        d.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r], fill=self.st.line_color)

    def _arrow(self, d: ImageDraw.ImageDraw, tip, direction) -> None:
        size = self.st.dot_radius * 2.6
        dx, dy = direction
        norm = math.hypot(dx, dy) or 1.0
        dx, dy = dx / norm, dy / norm
        px, py = -dy, dx
        back = (tip[0] - dx * size, tip[1] - dy * size)
        d.polygon([tip,
                   (back[0] + px * size * 0.45, back[1] + py * size * 0.45),
                   (back[0] - px * size * 0.45, back[1] - py * size * 0.45)],
                  fill=self.st.line_color)

    def _span(self, d: ImageDraw.ImageDraw, span) -> None:
        """Интервал поперёк детали: размерная линия с засечками на обеих кромках.

        Стрелка показывает только куда смотреть, интервал — что именно померено:
        от кромки до кромки. Размеру шинки выносная линия наружу не нужна, её
        сечение целиком лежит на металле и читается прямо на месте.

        Лежит он там же, где и блик, поэтому рисуется не общей линией разметки, а
        своей — темнее и толще, по светлой подложке. Серая нитка поперёк
        полированной шинки на кадре не читается: приёмка её попросту не находит.
        """
        (x0, y0), (x1, y1) = span
        dx, dy = x1 - x0, y1 - y0
        length = math.hypot(dx, dy) or 1.0
        px, py = -dy / length, dx / length            # нормаль к интервалу
        tick = self.st.band_tick
        strokes = [((x0, y0), (x1, y1))]
        for x, y in ((x0, y0), (x1, y1)):
            strokes.append(((x - px * tick, y - py * tick), (x + px * tick, y + py * tick)))
        halo = self.st.band_width + max(2, round(self.st.halo_grow))
        for p1, p2 in strokes:
            d.line([p1, p2], fill=self.st.halo_color, width=halo)
        for p1, p2 in strokes:
            self._line(d, p1, p2, color=self.st.band_color, width=self.st.band_width)

    def _paste_text(self, canvas: Image.Image, text: str, center, rotate: int = 0,
                    kind: str = "label"):
        f = self.label_font if kind == "label" else self.caption_font
        layer = _text_layer(text, f, self.st.text_color, rotate)
        x = round(center[0] - layer.width / 2)
        y = round(center[1] - layer.height / 2)
        canvas.alpha_composite(layer, (x, y))
        return (x, y, x + layer.width, y + layer.height)

    # -- габаритный размер ---------------------------------------------

    def _edge_at(self, dim: LinearDim, value: float, beyond: float,
                 product: Product | None) -> tuple[float, bool]:
        """Кромка измеряемого элемента под концом размера — откуда пойдёт выносная.

        Одна кромка на оба конца годится только ровному краю. На скруглении —
        а шинка скруглена вся — один конец повисает в воздухе, второй уезжает
        внутрь металла. Поэтому кромка под каждым концом ищется лучом со стороны
        размерной линии, отдельно.

        Ищется она **внутри рамки измеряемого элемента**, а не по всему силуэту.
        Луч по силуэту останавливается на первом металле, а первым по дороге к
        декоративному элементу лежит плечо шинки: выносная упиралась в него и
        прочерчивала полкадра поперёк изделия. Габарит меряет декоративный
        элемент — с ним линия и смыкается.
        """
        if product is None or not dim.snap or not dim.box:
            return dim.base, False
        bx0, by0, bx1, by1 = dim.box
        if dim.axis == "h":
            start, end = (value, by0 if dim.side < 0 else by1), (value, by1 if dim.side < 0 else by0)
        else:
            start, end = (bx0 if dim.side < 0 else bx1, value), (bx1 if dim.side < 0 else bx0, value)
        hit = product.ray_hit(start, end)
        if hit is None:
            return dim.base, False
        return (hit[1] if dim.axis == "h" else hit[0]), True

    def draw_linear(self, canvas: Image.Image, d: ImageDraw.ImageDraw, dim: LinearDim,
                    product: Product | None = None) -> list:
        st = self.st
        a, b = sorted((dim.a, dim.b))
        span = b - a
        rotate = 90 if dim.axis == "v" else 0
        tw, th = measure(dim.text, self.label_font)
        # вдоль измеряемой оси подпись занимает свою длину в обоих случаях:
        # вертикальная повёрнута на 90°, и её строка ложится вдоль Y
        text_extent = tw

        # выносные линии идут от размерной линии до кромки измеряемого элемента и
        # немного перехлёстывают размерную. Нашлась кромка — линия доходит до неё,
        # сколько бы это ни было: висящая в стороне засечка не показывает, что
        # именно померено. Не нашлась (столбец рамки пуст) — остаётся короткая
        # засечка длиной `ext_max`, как в эталонах.
        beyond = dim.level + st.tick_over * dim.side
        for value in (a, b):
            edge, found = self._edge_at(dim, value, beyond, product)
            edge += st.ext_gap * dim.side
            reach = (beyond - edge) * dim.side
            if reach < st.tick_over:
                edge = beyond - st.tick_over * dim.side
            elif not found or reach > st.ext_max * EXT_RUNAWAY:
                edge = beyond - st.ext_max * dim.side
            p1 = (value, edge) if dim.axis == "h" else (edge, value)
            p2 = (value, beyond) if dim.axis == "h" else (beyond, value)
            self._line(d, p1, p2)

        boxes = []
        inline = span >= text_extent + 2 * st.line_break + 2 * st.tick_over
        if inline:
            mid = (a + b) / 2
            half = text_extent / 2 + st.line_break
            for s0, s1 in ((a, mid - half), (mid + half, b)):
                if s1 - s0 <= 1:
                    continue
                p1 = (s0, dim.level) if dim.axis == "h" else (dim.level, s0)
                p2 = (s1, dim.level) if dim.axis == "h" else (dim.level, s1)
                self._line(d, p1, p2)
            center = (mid, dim.level) if dim.axis == "h" else (dim.level, mid)
            boxes.append(self._paste_text(canvas, dim.text, center, rotate))
        else:
            # короткий размер: линию не рвём, число ставим рядом и без поворота,
            # как в эталонных выкладках SOKOLOV
            p1 = (a, dim.level) if dim.axis == "h" else (dim.level, a)
            p2 = (b, dim.level) if dim.axis == "h" else (dim.level, b)
            self._line(d, p1, p2)
            gap = st.line_break + st.dot_radius
            if dim.axis == "h":
                tail = st.line_break * 1.6
                self._line(d, p2, (b + tail, dim.level))
                center = (b + tail + gap + tw / 2, dim.level)
            else:
                out = dim.side if dim.side else 1
                center = (dim.level + out * (gap + tw / 2), (a + b) / 2)
                if center[0] - tw / 2 < st.min_margin or center[0] + tw / 2 > self.size - st.min_margin:
                    center = (dim.level - out * (gap + tw / 2), (a + b) / 2)
            boxes.append(self._paste_text(canvas, dim.text, center))

        for value in (a, b):
            self._dot(d, (value, dim.level) if dim.axis == "h" else (dim.level, value))
        return boxes

    # -- наклонный габарит ----------------------------------------------

    def draw_inclined(self, canvas: Image.Image, d: ImageDraw.ImageDraw,
                      dim: InclinedDim, product: Product | None = None) -> list:
        st = self.st
        ax, ay = dim.a
        bx, by = dim.b
        nx, ny = dim.normal
        span = math.hypot(bx - ax, by - ay)
        if span < 1:
            return []
        ux, uy = (bx - ax) / span, (by - ay) / span
        tw, th = measure(dim.text, self.label_font)

        def shift(point, distance):
            return point[0] + nx * distance, point[1] + ny * distance

        # Выносная линия начинается на кромке изделия, а не на углу описанного
        # прямоугольника: у наклонного элемента угол приходится на пустое поле.
        far = dim.offset + st.tick_over
        for point in (dim.a, dim.b):
            start = shift(point, st.ext_gap)
            if product is not None:
                hit = product.ray_hit(shift(point, far), point, overshoot=dim.overshoot)
                if hit:
                    start = shift(hit, st.ext_gap)
            self._line(d, start, shift(point, far))

        start, end = shift(dim.a, dim.offset), shift(dim.b, dim.offset)
        middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)

        if span >= tw + 2 * st.line_break + 2 * st.tick_over:
            half = tw / 2 + st.line_break
            self._line(d, start, (middle[0] - ux * half, middle[1] - uy * half))
            self._line(d, (middle[0] + ux * half, middle[1] + uy * half), end)
            text_at, rotate = middle, _upright(ux, uy)
        else:
            # короткий размер: линию не рвём, число ставим за её концом
            self._line(d, start, end)
            tail = st.line_break * 1.6
            tip = (end[0] + ux * tail, end[1] + uy * tail)
            self._line(d, end, tip)
            gap = st.line_break + st.dot_radius + tw / 2
            text_at, rotate = (tip[0] + ux * gap, tip[1] + uy * gap), 0

        boxes = [self._paste_text(canvas, dim.text, text_at, rotate)]
        for point in (start, end):
            self._dot(d, point)
        return boxes

    # -- выноска --------------------------------------------------------

    def _fit_shelf(self, elbow_x: float, direction: int, length: float,
                   text_width: float) -> tuple[float, int]:
        """Подбирает конец полки так, чтобы число уместилось на холсте.

        Если с выбранной стороны места нет, выноска разворачивается в другую.
        """
        st = self.st
        need = st.line_break + text_width + TEXT_PAD * 2 + st.min_margin
        min_shelf = st.tick_over
        for sign in (direction, -direction):
            limit = (self.size - need) if sign > 0 else need
            far = elbow_x + sign * length
            far = min(far, limit) if sign > 0 else max(far, limit)
            if (far - elbow_x) * sign >= min_shelf:
                return far, sign
        return elbow_x + direction * min_shelf, direction

    def draw_leader(self, canvas: Image.Image, d: ImageDraw.ImageDraw, ld: Leader,
                    product: Product | None = None) -> list:
        st = self.st
        ax, ay = ld.anchor
        ex, ey = ld.elbow
        tw, th = measure(ld.text, self.label_font)
        far_x, direction = self._fit_shelf(ex, ld.direction, ld.length, tw)
        far = (far_x, ey)

        # Конец выноски сажаем ровно на силуэт: идём от излома к якорю и
        # останавливаемся на первом металле. Якорь, посчитанный по полосе строк,
        # на наклонной кромке отстоит от неё на величину наклона, и зазор виден.
        if product is not None and ld.snap:
            hit = product.ray_hit((ex, ey), (ax, ay), overshoot=ld.overshoot,
                                  within=ld.hit_box)
            if hit:
                ax, ay = hit

        if ld.span:
            self._span(d, ld.span)
        elif ld.arrow == "v":
            self._arrow(d, (ax, ay), (0, -1 if ay <= ey else 1))
        elif ld.arrow == "h":
            self._arrow(d, (ax, ay), (-direction, 0))
        self._line(d, (ax, ay), (ex, ey))
        self._line(d, (ex, ey), far)
        self._dot(d, far)

        cx = far_x + direction * (st.line_break + tw / 2)
        boxes = [self._paste_text(canvas, ld.text, (cx, ey))]
        if ld.caption:
            cw, _ = measure(ld.caption, self.caption_font)
            half = cw / 2 + TEXT_PAD
            # Подпись уходит по другую сторону полки от наклонного участка: иначе
            # длинная подпись, поджатая краем холста, ложится прямо на свою же линию.
            up = ay > ey
            caption_y = ey + (-st.caption_size * 2.1 if up else st.caption_size * 1.6)
            mid_x = ex + direction * (half + st.line_break * 0.5)
            mid_x = min(max(mid_x, half + st.min_margin), self.size - half - st.min_margin)
            boxes.append(self._paste_text(canvas, ld.caption, (mid_x, caption_y),
                                          kind="caption"))
        return boxes

    # -- лист -----------------------------------------------------------

    def render(self, sheet: Sheet) -> tuple[Image.Image, list, list]:
        """Возвращает кадр, рамки подписей (индекс, box) и отрезки (индекс, отрезок)."""
        canvas = sheet.product.canvas.convert("RGBA")
        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        boxes: list = []
        segments: list = []
        for index, item in enumerate(sheet.items):
            self.segments = []
            if isinstance(item, LinearDim):
                drawn = self.draw_linear(overlay, d, item, sheet.product)
            elif isinstance(item, InclinedDim):
                drawn = self.draw_inclined(overlay, d, item, sheet.product)
            elif isinstance(item, Leader):
                drawn = self.draw_leader(overlay, d, item, sheet.product)
            else:
                continue
            boxes += [(index, box) for box in drawn]
            segments += [(index, seg) for seg in self.segments]
        canvas.alpha_composite(overlay)
        return canvas.convert("RGB"), boxes, segments
