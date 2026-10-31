"""Поиск декоративного элемента по текстуре кадра.

Модель зрения даёт только грубую локализацию: она уверенно показывает, в какой
части кадра декор, но его границы и наклон ставит на глаз. Между тем сам декор
на студийном фото отличается от гладкого металла физически — камни, паве, эмаль,
ажур дают резкий локальный контраст, полированная шинка не даёт почти никакого.
Поэтому декор ищется по карте локального контраста, а из кандидатов выбирается
тот, чьи пропорции совпадают с табличными «длина x ширина».

Отсюда два выигрыша, которых не даёт осевая рамка:

- вытянутый декор, лежащий по диагонали (дорожка паве, надпись), меряется вдоль
  своей оси, а не по описанному прямоугольнику, который заметно больше самого
  элемента;
- выноска высоты садится на сам декор, а не на подвернувшуюся под луч шинку.

Отбор намеренно строгий: если ни один кандидат не подходит, модуль отвечает
`None`, и раскладка работает по прежней осевой рамке. Ошибиться молча хуже, чем
не сработать.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import Product

TEXTURE_WINDOW = 9        # окно локального контраста, px
TEXTURE_QUANTILE = 0.70   # какая доля металла считается гладкой
MIN_PART = 0.004          # минимальная доля куска от площади изделия
AREA_RANGE = (0.15, 3.0)  # во сколько раз кандидат может отличаться от рамки модели
CONTAIN = 0.80            # какая доля кандидата обязана лежать в рамке модели
BOX_MARGIN = 0.22         # на сколько рамка модели расширяется для этой проверки
ASPECT_TOL = 1.18         # допустимое расхождение пропорций с таблицей
SIZE_RANGE = (0.5, 2.0)   # во сколько раз кандидат может разойтись с размером по просвету кольца
MAX_PARTS = 6             # больше кусков — перебор сочетаний уже не окупается

INCLINE_MIN_RATIO = 1.6   # ниже этого элемент считается компактным
INCLINE_MIN_ANGLE = 12.0  # ниже этого наклон не отыгрывает усложнения разметки


@dataclass
class Decor:
    """Декоративный элемент: осевая рамка плюс собственные оси, если они найдены."""
    box: tuple                      # осевая рамка (x0, y0, x1, y1)
    center: tuple = (0.0, 0.0)
    axis: tuple = (1.0, 0.0)        # единичный вектор длинной стороны
    long: float = 0.0               # размер вдоль длинной стороны, px
    short: float = 0.0
    source: str = "box"             # 'texture' — найден по контрасту кадра
    angle: float = 0.0              # наклон длинной стороны к горизонтали, градусы

    @property
    def normal(self) -> tuple:
        return -self.axis[1], self.axis[0]

    def corner(self, toward: tuple, inset: float = 0.12) -> tuple:
        """Угол элемента, самый дальний в сторону `toward`, слегка вдвинутый внутрь.

        Выноска высоты цепляется именно за угол: у любой другой точки контура под
        луч раньше попадает шинка, подходящая к декору вплотную.
        """
        ux, uy = self.axis
        nx, ny = self.normal
        along = 1 if ux * toward[0] + uy * toward[1] >= 0 else -1
        across = 1 if nx * toward[0] + ny * toward[1] >= 0 else -1
        half_long = self.long / 2 * (1 - inset)
        half_short = self.short / 2 * (1 - inset)
        return (self.center[0] + ux * along * half_long + nx * across * half_short,
                self.center[1] + uy * along * half_long + ny * across * half_short)

    @property
    def inclined(self) -> bool:
        """Стоит ли мерить элемент вдоль его собственных осей, а не по кадру.

        Наклонными размерными линиями показываются только вытянутые элементы. Для
        компактных это пробовали — повёрнутая вставка тоже меряется описанной
        рамкой неверно, — и вышло хуже: две наклонные линии крест-накрест над
        мелким камнем читаются как перечёркнутый кадр. Раздутую рамку компактного
        элемента лечит `_axial_span` в `markup/layout.py`, не трогая вид разметки.
        """
        if self.source not in ("texture", "manual") or self.short <= 0:
            return False
        return (self.long / self.short >= INCLINE_MIN_RATIO
                and abs(self.angle) >= INCLINE_MIN_ANGLE)


def from_box(box: tuple) -> Decor:
    """Декор по осевой рамке — поведение без распознавания текстуры.

    Длинная ось ставится вдоль длинной стороны рамки: на отрисовку это не влияет,
    но редактор показывает рамку так, как она есть, а не повёрнутой на бок.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    tall = height > width
    return Decor(box=tuple(box), center=((x0 + x1) / 2, (y0 + y1) / 2),
                 axis=(0.0, 1.0) if tall else (1.0, 0.0),
                 long=max(width, height), short=min(width, height),
                 angle=90.0 if tall else 0.0)


# ---------------------------------------------------------------- карта контраста

def texture_mask(product: Product, quantile: float = TEXTURE_QUANTILE) -> np.ndarray:
    """Маска «декоративного» металла: где локальный контраст выше типичного."""
    gray = cv2.cvtColor(np.asarray(product.canvas), cv2.COLOR_RGB2GRAY).astype(np.float32)
    window = (TEXTURE_WINDOW, TEXTURE_WINDOW)
    mean = cv2.blur(gray, window)
    variance = cv2.blur(gray * gray, window) - mean * mean
    contrast = np.sqrt(np.maximum(variance, 0.0))

    metal = product.thin
    if not metal.any():
        return np.zeros_like(metal, np.uint8)
    hot = (metal & (contrast > np.percentile(contrast[metal], quantile * 100))).astype(np.uint8)
    hot = cv2.morphologyEx(hot, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return cv2.morphologyEx(hot, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))


def _oriented(mask: np.ndarray):
    """Ориентированный прямоугольник вокруг маски: центр, оси, длины, наклон."""
    points = cv2.findNonZero(mask)
    if points is None:
        return None
    corners = cv2.boxPoints(cv2.minAreaRect(points))
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    (a, b) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    long = math.dist(a, b)
    short = min(math.dist(*e) for e in edges)
    if long < 8 or short < 4:
        return None
    # приводим к обычным float: numpy-числа отсюда утекают в план и рушат json
    axis = (float((b[0] - a[0]) / long), float((b[1] - a[1]) / long))
    if axis[0] < 0:                      # длинная ось всегда смотрит вправо
        axis = (-axis[0], -axis[1])
    center = (float(corners[:, 0].mean()), float(corners[:, 1].mean()))
    return center, axis, float(long), float(short)


# ---------------------------------------------------------------- отбор кандидата

def detect(product: Product, box: tuple, length_mm: float, width_mm: float,
           mm_per_px: float = 0.0) -> Decor | None:
    """Ищет декоративный элемент внутри грубой рамки `box`.

    Возвращает `None`, если ни один кандидат не совпал с табличными пропорциями:
    молча подставить неверный элемент хуже, чем оставить прежнюю рамку.

    `mm_per_px` — масштаб кадра, посчитанный не по рамке, а по просвету кольца.
    Если он передан, кандидаты проверяются ещё и на абсолютный размер: у кольца
    с одной вставкой и дорожкой паве вдоль шинки пропорции 1:1 бывают и у вставки,
    и у всей дорожки, и по одному отношению сторон их не различить — а по размеру
    в миллиметрах различить можно. Допуск широкий (`SIZE_RANGE`): просвет на кадре
    сжат ракурсом, и мера эта грубая — она отсекает промах в разы, а не проценты.
    """
    long_mm, short_mm = max(length_mm, width_mm), min(length_mm, width_mm)
    if short_mm <= 0:
        return None
    target = long_mm / short_mm

    hot = texture_mask(product)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(hot, 8)
    if count < 2:
        return None
    floor = MIN_PART * product.thin.sum()
    parts = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= floor]
    if not parts:
        return None

    x0, y0, x1, y1 = box
    box_area = max(1.0, (x1 - x0) * (y1 - y0))
    mx, my = BOX_MARGIN * (x1 - x0), BOX_MARGIN * (y1 - y0)
    outer = (x0 - mx, y0 - my, x1 + mx, y1 + my)

    groups = [[i] for i in parts]
    if 1 < len(parts) <= MAX_PARTS:
        for size in (2, 3):
            groups += [list(c) for c in itertools.combinations(parts, size)]
    if len(parts) > 1:
        groups.append(parts)

    best = None
    for group in groups:
        mask = np.isin(labels, group).astype(np.uint8)
        total = int(mask.sum())
        if not total:
            continue
        shape = _oriented(mask)
        if shape is None:
            continue
        center, axis, long, short = shape
        if not AREA_RANGE[0] <= long * short / box_area <= AREA_RANGE[1]:
            continue
        if mm_per_px > 0:
            want = long_mm / mm_per_px
            if not SIZE_RANGE[0] <= long / want <= SIZE_RANGE[1]:
                continue
        if not (x0 <= center[0] <= x1 and y0 <= center[1] <= y1):
            continue
        inside = int(mask[max(0, int(outer[1])):int(outer[3]),
                          max(0, int(outer[0])):int(outer[2])].sum())
        if inside / total < CONTAIN:
            continue
        miss = abs(math.log((long / short) / target))
        if miss > math.log(ASPECT_TOL):
            continue
        if best is None or miss < best[0]:
            ys, xs = np.where(mask)
            best = (miss, Decor(
                box=(float(xs.min()), float(ys.min()),
                     float(xs.max()) + 1, float(ys.max()) + 1),
                center=center, axis=axis, long=long, short=short,
                source="texture", angle=math.degrees(math.atan2(axis[1], axis[0])),
            ))
    return best[1] if best else None
