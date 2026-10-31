"""Масштаб кадра: сколько миллиметров приходится на пиксель.

Масштаб нужен всей разметке: по нему решается, можно ли вообще показать размер
габаритом, какой длины ждать сечение шинки и не промахнулась ли рамка
декоративного элемента. До сих пор он выводился из одной-единственной меры —
рамки декоративного элемента, — а рамка сама себе не свидетель: раздутая вдвое,
она вдвое же врёт про масштаб, и проверить это было нечем.

Поэтому мер здесь две, и они независимы.

**По рамке.** Табличные «длина x ширина» делятся на стороны рамки. Мера точная,
когда рамка стоит верно, и произвольно неверная, когда нет.

**По кольцу.** Кольцо всегда размером с палец: внутренний диаметр 15,5–20,5 мм,
медиана по выборке 17,3. Изделие вписывается в холст по большей стороне, и у
кольца эта сторона — наружный диаметр (большая ось эллипса на любом ракурсе не
сокращается). Отсюда масштаб, не зависящий ни от рамки, ни от модели зрения.
Когда в таблице стоит размер кольца — «длина» в 30 мм и больше, это обхват, а не
габарит, — внутренний диаметр берётся из него, и мера становится точной.

На контрольной выборке из 75 колец, где обе меры сходятся между собой, отношение
«рамка / кольцо» имеет медиану 0,96: мера по кольцу несмещённая. Разброс у неё
широкий (децили 0,70 и 1,35) — это цена того, что размер кольца в таблице обычно
не указан. Поэтому она не уточняет масштаб, а ловит промах в разы: рамку,
севшую на пол-изделия, и число, которое габаритом не показать.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from .imaging import Product

BORE_TYPICAL = 17.3      # мм: медиана внутреннего диаметра кольца по выборке
RING_SIZE_MIN = 30.0     # мм: «длина» больше этой — размер кольца (обхват), а не габарит
BORE_RANGE = (12.0, 24.0)   # мм: правдоподобный внутренний диаметр кольца
#: во сколько раз мерам позволено разойтись, прежде чем рамка считается неверной
DISAGREE = 1.8
COS_MIN = 0.12           # ниже этого кольцо снято практически с торца


@dataclass(frozen=True)
class Scale:
    """Масштаб кадра и то, чем он подтверждён."""
    mm_per_px: float          # рабочая мера
    by_decor: float           # по рамке декоративного элемента
    by_ring: float            # по кольцу, независимо от рамки
    cos_tilt: float           # косинус наклона плоскости кольца к плоскости кадра
    source: str               # 'согласие' | 'кольцо' | 'рамка' | 'нет'
    spread: float             # во сколько раз меры разошлись (1.0 — сошлись)

    @property
    def ok(self) -> bool:
        return self.mm_per_px > 0

    def px(self, mm: float) -> float:
        """Табличный размер в пикселях этого кадра."""
        return mm / self.mm_per_px if self.mm_per_px else 0.0

    @property
    def sin_tilt(self) -> float:
        return math.sqrt(max(0.0, 1.0 - self.cos_tilt ** 2))

    @property
    def decor_trusted(self) -> bool:
        """Можно ли верить рамке: её мера подтверждена кольцом или проверить нечем."""
        return self.by_decor > 0 and (self.by_ring <= 0 or self.spread <= DISAGREE)


def ring_size_mm(row) -> float:
    """Размер кольца из таблицы, если «длина» — это он, а не габарит.

    В рабочей таблице у колец с декором на всю шинку (обручальные, цепи, дорожки)
    в графе «длина декоративного элемента» стоит размер кольца: 54 мм — это обхват
    пальца, а не что-то, что можно померить на кадре. Отличается он от настоящего
    габарита порядком: декоративные элементы выборки укладываются в 29 мм.
    """
    return float(row.length) if float(row.length or 0) >= RING_SIZE_MIN else 0.0


def band_twins(row) -> set:
    """Размеры декоративного элемента, которые на деле являются размерами шинки.

    У кольца, декор которого идёт по всей шинке — обручального, цепи, дорожки, —
    в графе «ширина декоративного элемента» стоит ширина самой шинки, тот же
    миллиметр, что и в графе шинки. Это не совпадение и не ошибка таблицы: декор
    там и есть шинка.

    Такой размер идёт вдоль оси кольца, то есть вглубь кадра, и на кадре виден
    сокращённым на ракурс. Ни габаритом его не показать, ни масштаб по нему не
    вывести: рамка меряет проекцию, а в таблице стоит натура.
    """
    twins = set()
    band = float(getattr(row, "band_w", 0) or 0)
    if band <= 0:
        return twins
    for key in ("width", "length"):
        value = float(getattr(row, key, 0) or 0)
        if value > 0 and abs(value - band) < 1e-6:
            twins.add(key)
    return twins


def table_dims(row) -> dict:
    """Табличные размеры, годные для промера кадра.

    Вычёркиваются два вида чисел. Размер кольца из графы «длина» — обхват пальца
    на кадре не лежит нигде, и мерить им рамку значит получить масштаб, ошибочный
    втрое. И размеры-двойники шинки — они лежат вдоль оси кольца и на кадре
    сокращены ракурсом, а в таблице записаны натурой.
    """
    values = dict(row.dims())
    if ring_size_mm(row):
        values["length"] = 0.0
    for key in band_twins(row):
        values[key] = 0.0
    return values


def bore_mm(row) -> float:
    """Внутренний диаметр кольца: из размера, а без него — по медиане выборки."""
    size = ring_size_mm(row)
    return size / math.pi if size else BORE_TYPICAL


def caliper(product: Product) -> float:
    """Наибольший габарит изделия на кадре, px."""
    x0, y0, x1, y1 = product.bbox
    return float(max(x1 - x0, y1 - y0))


@dataclass(frozen=True)
class Ellipse:
    """Кольцо на кадре: эллипс, в который укладывается его силуэт."""
    center: tuple
    major: float           # большая ось, px — наружный диаметр при любом ракурсе
    minor: float           # малая ось, px — он же, сокращённый на наклон
    umajor: tuple          # единичное направление большой оси
    uminor: tuple          # единичное направление малой оси

    @property
    def cos_tilt(self) -> float:
        return min(1.0, max(COS_MIN, self.minor / self.major)) if self.major > 1 else 1.0


def ellipse(product: Product) -> Ellipse | None:
    """Эллипс силуэта: центр, оси и их направления.

    Кольцо на кадре — эллипс: большая ось равна наружному диаметру при любом
    ракурсе, малая сокращается на косинус наклона. Отсюда и ракурс, и — что важнее
    для промера шинки — направления осей.

    Ширина шинки идёт вдоль оси кольца, а та на кадре смотрит вдоль малой оси
    эллипса. На концах большой оси ширина поэтому вырождается: там шинка повёрнута
    к камере торцом, её ход и её ширина на кадре совпадают по направлению, и
    сечение поперёк хода меряет уже не ширину. На концах малой оси, наоборот,
    ширина видна целиком и поперёк хода — там её и надо мерить.

    Считается по внешнему контуру залитого силуэта, поэтому ажур и камни на нём
    почти не сказываются.
    """
    cached = product.__dict__.get("_silhouette_ellipse")
    if cached is not None:
        return None if cached is False else cached
    found = None
    contours, _ = cv2.findContours(product.solid.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        if len(biggest) >= 5:
            (cx, cy), (d1, d2), angle = cv2.fitEllipse(biggest)
            radians = math.radians(angle)
            # cv2 отдаёт (ширина, высота) повёрнутого прямоугольника и его угол:
            # ширина лежит вдоль (cos, sin), высота — вдоль нормали к ней
            u1 = (math.cos(radians), math.sin(radians))
            u2 = (-math.sin(radians), math.cos(radians))
            (major, umajor), (minor, uminor) = (
                ((d1, u1), (d2, u2)) if d1 >= d2 else ((d2, u2), (d1, u1)))
            if major > 1:
                found = Ellipse((float(cx), float(cy)), float(major), float(minor),
                                umajor, uminor)
    product.__dict__["_silhouette_ellipse"] = found if found is not None else False
    return found


def cos_tilt(product: Product, row=None) -> float:
    """Косинус наклона плоскости кольца к плоскости кадра.

    Отношение осей силуэта берётся с поправкой на ширину шинки. Силуэт кольца —
    это не его средняя окружность: шинка имеет ширину, и она у кольца, стоящего
    под наклоном, добавляется к малой оси. Большая ось остаётся диаметром при
    любом ракурсе, а малая выходит `D·cosθ + w·sinθ`.

    Без поправки широкая шинка выдаёт себя за ровный ракурс: у кольца с шинкой в
    11 мм отношение осей 0,90 читалось как наклон 25°, тогда как на деле кольцо
    наклонено на 64° и показывает шинку почти целиком. Отсюда и метка ширины
    вдвое короче видимой грани — по 25° кадр «должен» показывать лишь её
    половину.

    Уравнение `cosθ + k·sinθ = отношение` имеет два корня; берётся больший — тот,
    на котором наклон увеличивает сокращение. Меньший отвечает кольцу, лежащему
    почти плашмя, а такой ракурс на студийных фото не встречается: изделие
    снимают так, чтобы показать лицевую грань.
    """
    shape = ellipse(product)
    if shape is None:
        return 1.0
    ratio = shape.cos_tilt
    if row is None:
        return ratio
    width = float(getattr(row, "band_w", 0) or 0)
    outer = bore_mm(row) + 2 * max(0.5, float(getattr(row, "band_t", 0) or 0))
    if width <= 0 or outer <= 0:
        return ratio
    return _solve_tilt(ratio, width / outer)


def _solve_tilt(ratio: float, k: float) -> float:
    """Косинус θ из уравнения cosθ + k·sinθ = ratio, по нисходящей ветви.

    Ветвь начинается за максимумом выражения — при θ = arctg k. Если измеренное
    отношение выше максимума (кольцо снято ровнее, чем допускает ширина шинки, —
    либо силуэт неточен), возвращается сам максимум.
    """
    peak = math.atan(k)
    best, miss = peak, None
    steps = 180
    for step in range(steps + 1):
        theta = peak + (math.pi / 2 - peak) * step / steps
        value = math.cos(theta) + k * math.sin(theta)
        error = abs(value - ratio)
        if miss is None or error < miss:
            best, miss = theta, error
    return max(COS_MIN, min(1.0, math.cos(best)))


def by_ring(product: Product, plan, row) -> float:
    """Масштаб по кольцу, мм на пиксель. 0 — изделие не кольцо."""
    if getattr(plan, "item_type", "") != "кольцо":
        return 0.0
    outer = bore_mm(row) + 2 * max(0.5, float(row.band_t or 0))
    span = caliper(product)
    return outer / span if span > 1 else 0.0


def by_decor(plan, values: dict, element) -> tuple[float, float]:
    """Масштаб по рамке декоративного элемента: (мм на пиксель, внутренний разброс).

    Сторон у рамки две, и каждая даёт свою меру. Пока они сходятся, рамка стоит
    верно и среднее по ним — самая точная мера кадра. Разошлись — рамка меряет
    не то, что записано в таблице, и усреднять их нельзя: два промаха в разные
    стороны дают в среднем правдоподобное число, за которым не видно, что ни
    одна сторона не сошлась. Такую меру возвращаем с её разбросом, а решает по
    нему `measure`.
    """
    if element.source in ("texture", "manual"):
        # у элемента известны собственные оси: длинная сторона отвечает большему
        # табличному размеру, и масштаб берётся по ней
        long_mm = max(values.get("length", 0.0), values.get("width", 0.0))
        short_mm = min(values.get("length", 0.0), values.get("width", 0.0))
        pairs = [(long_mm, element.long), (short_mm, element.short)]
    else:
        dx0, dy0, dx1, dy1 = element.box
        pairs = [(values.get(plan.axes.get("horizontal"), 0.0), dx1 - dx0),
                 (values.get(plan.axes.get("vertical"), 0.0), dy1 - dy0)]
    samples = [mm / size for mm, size in pairs if mm > 0 and size > 4]
    if not samples:
        return 0.0, 1.0
    spread = max(samples) / min(samples) if min(samples) > 0 else float("inf")
    return sum(samples) / len(samples), spread


def measure(product: Product, plan, row, element) -> Scale:
    """Сводит обе меры в один масштаб кадра.

    Сошлись — берём среднее: рамка мельче и точнее, кольцо её удерживает.
    Разошлись в разы — верим кольцу: оно не зависит ни от рамки, ни от ответа
    модели. Кольца нет — остаётся рамка, и проверить её нечем.

    Рамка, у которой не сходятся собственные стороны, в расчёт не идёт вовсе:
    она меряет не то, что записано в таблице, и её среднее — случайное число.
    """
    decor, inner = by_decor(plan, table_dims(row), element)
    ring = by_ring(product, plan, row)
    tilt = cos_tilt(product, row) if ring else 1.0
    if inner > DISAGREE:
        decor = 0.0
    if decor > 0 and ring > 0:
        spread = max(decor / ring, ring / decor)
        if spread <= DISAGREE:
            return Scale((decor + ring) / 2, decor, ring, tilt, "согласие", spread)
        return Scale(ring, decor, ring, tilt, "кольцо", spread)
    if decor > 0:
        return Scale(decor, decor, 0.0, tilt, "рамка", 1.0)
    if ring > 0:
        return Scale(ring, 0.0, ring, tilt, "кольцо", 1.0)
    return Scale(0.0, 0.0, 0.0, tilt, "нет", 1.0)


def bore_check(product: Product, scale: Scale, plan) -> str:
    """Замечание, если по масштабу кадра просвет кольца выходит неправдоподобным.

    Смысл замечания — уличить рамку декоративного элемента, поэтому спрашивается
    оно только там, где масштаб от рамки и зависит. Масштаб, взятый по кольцу,
    рамке ничем не обязан, и проверять его просветом — значит проверять кольцо
    кольцом: у ажурного изделия просвет частью закрыт своими же деталями, и
    замечание срабатывало бы на ровном месте.
    """
    from .layout import _hole

    if getattr(plan, "item_type", "") != "кольцо" or not scale.ok:
        return ""
    if scale.source == "кольцо":
        return ""
    hole = _hole(product)
    if hole is None:
        return ""
    xs = np.nonzero(hole)[1]
    bore = (int(xs.max()) - int(xs.min())) * scale.mm_per_px
    low, high = BORE_RANGE
    if low <= bore <= high:
        return ""
    return (f"масштаб кадра сомнителен: по нему просвет кольца {bore:.0f} мм — "
            "проверьте рамку декоративного элемента")
