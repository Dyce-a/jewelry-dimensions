"""Раскладка: план + геометрия -> конкретные размерные линии и выноски."""
from __future__ import annotations

import math

import cv2
import numpy as np

from . import decor, scale
from .config import BAND_SPANS, DIMENSIONS, STYLE
from .imaging import Product
from .planner import Plan
from .render import InclinedDim, Leader, LinearDim, Renderer, Sheet, fmt

CAPTIONS = dict(DIMENSIONS)
MAX_PASSES = 5
MIN_BAND_PX = STYLE.offset_1 * 0.25     # короче этого интервал шинки не отложить по таблице
BAND_FLANK = 0.12                       # запасной бок: отступ от правого края, доля ширины
BAND_WINDOW = 0.045                     # окно, в котором определяется ход шинки, доля холста
HOLE_MIN = 0.004                        # доля кадра: дырки мельче — это блики, а не просвет
BAND_INSET = 0.77                       # запасная доля сечения, когда шинка в таблице не задана
BAND_INSET_MIN = 0.45                   # меньше этой доли интервал уже не читается как промер
BAND_INSET_MAX = 0.92                   # больше — упирается в обе кромки
#: Насколько промер может отойти от табличной толщины и всё ещё считаться ею.
#: Совпавший с толщиной промер — признак того, что кольцо снято в фас: ширина
#: шинки смотрит в камеру, и весь отрезок маски занимает толщина. Интервал там
#: мерил бы толщину, а подписан был бы шириной, — размер уходит на выноску.
BAND_AS_THICKNESS = (0.7, 1.4)
FLANK_BIN = 0.15                        # каким разбросом сечения считаются одним и тем же местом шинки
FLANK_SKEW = 35                         # на сколько градусов сечению позволено отойти от радиуса кольца
FLANK_FLOOR = 0.4                       # доля от девятого дециля: короче — проба снята не с шинки
FLANK_SWALLOW = 0.7                     # доля силуэта: рамка крупнее уже не фильтр, а всё изделие
BAND_APART = 0.11                       # доля холста: ближе этого метки шинки сходятся в кашу
FLANK_SAMPLES = 60                      # сколько проб сечения снимается с контура просвета
PROBE_FAN = 10                          # сколько лучей в каждую сторону от малой оси
PROBE_STEP = 3.0                        # шаг веера лучей, градусы
PROBE_MIN = 4                           # px: пробег короче — это блик, а не шинка
PROBE_ALARM = 2.5                       # во сколько раз промер может разойтись с ожиданием
#: На сколько градусов дуги разводятся две метки шинки. Порог высокий не для
#: красоты: перпендикулярно ходу шинки луч из центра идёт только у концов малой
#: оси, а между ними он режет ободок наискось и меряет длиннее правды. Две метки
#: поэтому либо стоят на разных концах оси, либо второй не будет вовсе.
PROBE_APART = 70
#: Долю подобрали по кадрам, принятым технологом: интервал должен кончаться на
#: кромке лицевой грани, а не на наружном контуре. Ярче/темнее эту кромку не
#: найти — яркость поперёк сечения ровная, — поэтому она задана долей.
BAND_ALARM = 4.0                        # расхождение, при котором промер уже не спишешь на ракурс
#: Во сколько раз габарит может разойтись со своим числом, прежде чем перестанет
#: быть габаритом. Порог широкий: независимая мера кадра — по кольцу — знает
#: размер пальца лишь с точностью до децилей 0,70–1,35, и придираться к процентам
#: ею нельзя. Зато промах в разы она ловит уверенно, а именно он и портит
#: выкладку: линия длиной в полкадра, подписанная «3 мм».
GABARIT_TOL = 1.8
#: Насколько кандидат обязан выиграть у закреплённой рамки, чтобы её отменить.
DECOR_PICK = 1.15
TAIL_LIMIT = 0.18                       # насколько габарит идёт за рамку по хвосту, доля стороны
TAIL_JOIN = 10                          # px: на сколько пробег металла смещается за шаг и остаётся собой
TAIL_SLACK = 2                          # px: на столько пробегу позволено потолстеть — это шум маски, а не деталь
TAIL_THIN = 0.35                        # каким тонким хвост обязан быть уже на кромке, доля поперечника рамки


# ---------------------------------------------------------------- геометрия

def _clip_box(box, bbox):
    x0, y0, x1, y1 = box
    bx0, by0, bx1, by1 = bbox
    x0, x1 = max(x0, bx0), min(x1, bx1)
    y0, y1 = max(y0, by0), min(y1, by1)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return bbox
    return x0, y0, x1, y1


def _bottom_run(product: Product, x: int) -> tuple[int, int] | None:
    """Нижний непрерывный отрезок изделия в колонке x — сечение шинки."""
    x = int(np.clip(x, 0, product.size - 1))
    column = product.thin[:, x]
    ys = np.where(column)[0]
    if not len(ys):
        return None
    end = int(ys[-1])
    start = end
    while start > 0 and column[start - 1]:
        start -= 1
    return start, end


def _bottom_column(product: Product) -> int:
    """Колонка самой нижней точки изделия — там шинка режется поперёк.

    Берётся середина самой длинной площадки в нижних строках маски: у изделия,
    попавшего в кадр двумя ракурсами, нижних площадок несколько, и середина всех
    сразу пришлась бы на пустоту между ними.
    """
    px0, _, px1, py1 = product.bbox
    strip = product.thin[max(0, py1 - 3):py1, :].any(axis=0)
    xs = np.where(strip)[0]
    if not len(xs):
        return int((px0 + px1) / 2)
    runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    return int(round(float(max(runs, key=len).mean())))


def _band_column(product: Product, point) -> int:
    """Колонка, в которой режем шинку по ширине.

    По умолчанию — нижняя точка изделия. Но место промера можно поправить руками в
    редакторе, и отметка человека на кадре точнее любой эвристики. Принимаем её,
    если она и правда стоит у низа изделия: отметка с бока даёт косой разрез —
    там шинка уходит вглубь кадра, и ширина по нему выходит вдвое длиннее.
    """
    px0, py0, px1, py1 = product.bbox
    bottom = _bottom_column(product)
    if not point:
        return bottom
    column = int(np.clip(point[0], px0, px1 - 1))
    run = _bottom_run(product, column)
    if run and py1 - run[1] <= 0.15 * (py1 - py0):
        return column
    return bottom


def _section(product: Product, column: int) -> tuple[float, float] | None:
    """Сечение шинки в колонке: внутренняя и наружная кромки."""
    run = _bottom_run(product, column)
    return (float(run[0]), float(run[1] + 1)) if run else None


def _cut(product: Product, point, axis: str) -> tuple[float, float] | None:
    """Непрерывный отрезок металла через точку — разрез шинки по оси кадра.

    Осталось для редактора: раскладка режет шинку поперёк её хода
    (`_band_section`), а редактору нужен и осевой разрез — им человек проверяет,
    что отметка промера стоит на шинке, а не на плече.

    Точка ставится на шинку, а границы отрезка ищутся по маске в обе стороны:
    интервал получается ровно по кромкам, а не по рамке и не по отступу.
    """
    x = int(np.clip(point[0], 0, product.size - 1))
    y = int(np.clip(point[1], 0, product.size - 1))
    line = product.thin[:, x] if axis == "v" else product.thin[y]
    index = y if axis == "v" else x
    if not line[index]:
        metal = np.where(line)[0]
        if not len(metal):
            return None
        index = int(metal[np.argmin(np.abs(metal - index))])
    start = end = index
    while start > 0 and line[start - 1]:
        start -= 1
    while end + 1 < len(line) and line[end + 1]:
        end += 1
    return float(start), float(end + 1)


def _hole(product: Product):
    """Просвет кольца: самая большая область фона, замкнутая металлом.

    Просвет — единственное место кадра, которое прямо показывает внутреннюю
    кромку шинки. Без него «бок» изделия приходилось брать отступом от рамки, а
    там шинка идёт наискось, и разрез по оси кадра выходил в полтора-два раза
    длиннее её сечения.
    """
    cached = product.__dict__.get("_ring_hole")
    if cached is not None:
        return None if cached is False else cached
    count, labels = cv2.connectedComponents((~product.thin).astype(np.uint8), connectivity=4)
    outside = set(np.unique(labels[0])) | set(np.unique(labels[-1]))
    outside |= set(np.unique(labels[:, 0])) | set(np.unique(labels[:, -1]))
    best, area = None, HOLE_MIN * product.thin.size
    for label in range(1, count):
        if label in outside:
            continue
        mask = labels == label
        size = int(mask.sum())
        if size > area:
            best, area = mask, size
    product.__dict__["_ring_hole"] = best if best is not None else False
    return best


def _ring_center(product: Product) -> tuple[float, float]:
    """Центр кольца: центр просвета, а без просвета — центр силуэта.

    Считается по всей маске кадра, а спрашивают его в цикле — на каждое сечение
    шинки, — поэтому ответ держим при изделии.
    """
    cached = product.__dict__.get("_ring_center")
    if cached is not None:
        return cached
    hole = _hole(product)
    if hole is not None:
        ys, xs = np.nonzero(hole)
        center = (float(xs.mean()), float(ys.mean()))
    else:
        px0, py0, px1, py1 = product.bbox
        center = ((px0 + px1) / 2, (py0 + py1) / 2)
    product.__dict__["_ring_center"] = center
    return center


def _band_axis(product: Product, point) -> tuple[float, float]:
    """Направление, вдоль которого идёт шинка в этой точке — главная ось маски.

    Разрез по оси кадра годится только там, где шинка идёт строго поперёк этой
    оси. На боку кольца она идёт наискось, и осевой разрез растягивается на
    1/sin наклона: отсюда и брались одинаковые с виду толщина и ширина.
    """
    radius = max(6, int(BAND_WINDOW * product.size))
    x, y = int(point[0]), int(point[1])
    window = product.thin[max(0, y - radius):y + radius + 1,
                          max(0, x - radius):x + radius + 1]
    ys, xs = np.nonzero(window)
    if len(xs) < 12:
        return 1.0, 0.0
    cloud = np.stack([xs - xs.mean(), ys - ys.mean()]).astype(np.float64)
    _, vectors = np.linalg.eigh(cloud @ cloud.T)
    return float(vectors[0, -1]), float(vectors[1, -1])


def _band_section(product: Product, point) -> tuple[tuple, tuple] | None:
    """Сечение шинки поперёк её хода: (наружная кромка, внутренняя кромка).

    Наружной считается та, что дальше от центра кольца, — от неё и откладываются
    размеры, потому что наружный контур виден на кадре всегда, а внутренний
    прячется за самой шинкой на закрытых ракурсах.
    """
    ux, uy = _band_axis(product, point)
    nx, ny = -uy, ux
    limit = int(product.size * 0.3)
    ends = []
    for sign in (1, -1):
        x, y = float(point[0]), float(point[1])
        last = (x, y)
        for _ in range(limit * 2):
            x, y = x + sign * nx * 0.5, y + sign * ny * 0.5
            if not (0 <= int(x) < product.size and 0 <= int(y) < product.size):
                break
            if not product.thin[int(y), int(x)]:
                break
            last = (x, y)
        ends.append(last)
    ends.sort(key=lambda end: -math.dist(end, _ring_center(product)))
    return (ends[0], ends[1]) if math.dist(ends[0], ends[1]) >= 2 else None


def _bore_scale(product: Product, plan: Plan, row) -> float:
    """Масштаб кадра по кольцу, мм на пиксель — мера, не зависящая от рамки.

    Оставлена отдельной функцией потому, что её спрашивает поиск декоративного
    элемента: кандидату мало совпасть пропорциями, он обязан ещё и попасть в
    размер. Считает `markup/scale.py`.
    """
    return scale.by_ring(product, plan, row)


def _lay(start, toward, want: float) -> tuple:
    """Отрезок длиной `want` от точки `start` в сторону `toward`."""
    length = math.dist(start, toward)
    want = min(want, length)
    ux = (toward[0] - start[0]) / length
    uy = (toward[1] - start[1]) / length
    return start, (start[0] + ux * want, start[1] + uy * want)


def _face_share(frame: scale.Scale, values: dict) -> float:
    """Какая доля сечения шинки — её ширина, а не толщина.

    Сечение на нижней дуге кольца — это не только ширина. Радиус кольца там идёт
    вглубь кадра под тем же наклоном, что и ось, поэтому в один и тот же отрезок
    маски укладываются обе величины: толщина, сокращённая на косинус наклона, и
    ширина, сокращённая на синус. Отсюда и жалоба, что метка ширины забирает
    больше, чем ширина: она забирала всё сечение.

    Доли считаются по таблице и наклону кадра, а не подбираются: на кольце с
    широкой шинкой ширина занимает девять десятых сечения, на тонком ободке — от
    силы половину. Прежняя постоянная доля (0,77) была средним по выборке и на
    обоих концах врала.
    """
    thickness = float(values.get("band_t") or 0.0)
    width = float(values.get("band_w") or 0.0)
    if thickness <= 0 or width <= 0:
        return BAND_INSET
    face = width * frame.sin_tilt
    total = thickness * frame.cos_tilt + face
    if total <= 0:
        return BAND_INSET
    return min(BAND_INSET_MAX, max(BAND_INSET_MIN, face / total))


def _band_interval(outer, inner, share: float) -> tuple:
    """Интервал размера на сечении шинки — по лицевой грани, от наружной кромки.

    Раньше длина откладывалась от внутренней кромки по таблице в масштабе кадра.
    Так метка вставала короче видимой шинки и, что хуже, ложилась не на ту её
    часть: со стороны просвета на нижней дуге лежит не лицевая грань, а внутренняя
    поверхность кольца, и на кадрах, снятых сверху, интервал уходил на неё.

    Потом интервал стал садиться на всё сечение по центру — и начал забирать
    вместе с гранью кусок внутренней поверхности, потому что сечение шире грани
    ровно на толщину, сокращённую ракурсом.

    Теперь он идёт от наружной кромки внутрь ровно на долю грани (`_face_share`):
    наружная кромка на кадре видна всегда, а грань лежит сразу за ней. Число
    рядом по-прежнему из таблицы — метка отвечает за «что померено», таблица за
    «сколько это в миллиметрах».
    """
    length = math.dist(outer, inner)
    if length < 1:
        return outer, inner
    ux, uy = (inner[0] - outer[0]) / length, (inner[1] - outer[1]) / length
    # от самой кромки интервал слегка отступает: упёршийся в неё, он читается как
    # промер кромки, а не металла
    gap = length * 0.04
    start = (outer[0] + ux * gap, outer[1] + uy * gap)
    reach = min(length - gap, length * share)
    return start, (outer[0] + ux * (gap + reach), outer[1] + uy * (gap + reach))


def _flank_candidates(product: Product, decor_box=None) -> tuple[list, bool]:
    """Сечения шинки по правой стороне кольца: ([(длина, точка), ...], чистая ли выборка).

    Идём по внутренней кромке — контуру просвета: он весь принадлежит шинке, и
    от него нормаль наружу всегда попадает в металл. Правой стороной выбор
    ограничен потому, что оттуда полку выноски видно, а через изделие её вести
    некуда.
    """
    cached = product.__dict__.get("_flank_candidates")
    if cached is not None:
        return cached
    hole = _hole(product)
    if hole is None:
        return [], False
    cx, cy = _ring_center(product)
    border = hole & ~cv2.erode(hole.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ys, xs = np.nonzero(border)
    right = xs >= cx
    xs, ys = xs[right], ys[right]
    if not len(xs):
        return [], False
    # сечение в точке стоит недёшево (главная ось маски в окне), а раскладка
    # пересобирается на каждом проходе: контур прореживаем до полусотни проб
    step = max(1, len(xs) // FLANK_SAMPLES)
    reach = max(4, int(BAND_WINDOW * product.size))
    inside = []
    outside = []
    for x, y in zip(xs[::step], ys[::step]):
        vx, vy = x - cx, y - cy
        norm = math.hypot(vx, vy) or 1.0
        vx, vy = vx / norm, vy / norm
        for step in range(2, reach):
            fx, fy = x + vx * step, y + vy * step
            if not (0 <= int(fx) < product.size and 0 <= int(fy) < product.size):
                break
            if not product.thin[int(fy), int(fx)]:
                continue
            section = _band_section(product, (fx, fy))
            # сечение обязано идти поперёк шинки, то есть вдоль радиуса кольца.
            # Косое — это не промер, а срез наискось: он длиннее самой шинки во
            # столько раз, во сколько ошибся угол
            if section and _radial(section, (fx, fy), (cx, cy)) <= FLANK_SKEW:
                sample = (math.dist(*section), (float(fx), float(fy)))
                (inside if decor_box and _within(decor_box, fx, fy) else outside).append(sample)
            break
    # рамка декоративного элемента накрывает каст с плечами: сечение там — не шинка.
    # Но она же бывает раздутой на пол-изделия, и тогда отбрасывать по ней нечего:
    # выборка идёт в дело целиком и помечается грязной — вместе с кастом в неё
    # попали сечения втрое длиннее шинки
    # рамка, накрывшая изделие почти целиком, фильтром быть перестала: отбрасывать
    # по ней нечего, и пробы из-под неё идут в дело как есть. Рамка поменьше —
    # фильтр рабочий, и всё, что под ней, это декор, а не шинка
    px0, py0, px1, py1 = product.bbox
    swallow = bool(decor_box) and (
        (decor_box[2] - decor_box[0]) >= FLANK_SWALLOW * (px1 - px0)
        and (decor_box[3] - decor_box[1]) >= FLANK_SWALLOW * (py1 - py0))
    pool = _solid(outside) or (_solid(inside) if swallow or not decor_box else [])
    found = (pool, bool(outside))
    product.__dict__["_flank_candidates"] = found
    return found


def _solid(samples: list) -> list:
    """Отсеивает пробы, снятые не с шинки: слишком короткие, чтобы быть ею.

    Контур просвета местами идёт вдоль шинки, а не поперёк, и луч оттуда упирается
    в фон через считаные пиксели. На тонком кольце таких проб набирается половина
    выборки — и самой населённой группой сечений оказываются они, а не металл.
    """
    if not samples:
        return samples
    top = float(np.percentile([length for length, _ in samples], 90))
    return [s for s in samples if s[0] >= top * FLANK_FLOOR]


def _within(box, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def _radial(section, point, center) -> float:
    """Угол между сечением и радиусом кольца в этой точке, в градусах."""
    (ox, oy), (ix, iy) = section
    vx, vy = ox - ix, oy - iy
    rx, ry = point[0] - center[0], point[1] - center[1]
    norm = math.hypot(vx, vy) * math.hypot(rx, ry)
    if norm < 1e-9:
        return 90.0
    return math.degrees(math.acos(min(1.0, abs(vx * rx + vy * ry) / norm)))


def _box_gap(box, point) -> float:
    """Насколько точка отстоит от рамки: 0 внутри, дальше — расстояние до неё."""
    if not box:
        return 0.0
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def _typical(lengths) -> float:
    """Самое населённое сечение выборки — ровный участок шинки.

    Медиана тут не годится: половина проб контура приходится на места, где шинка
    уходит под каст или показана в торец, и медиана садится между двумя разными
    вещами. Шинка же на большей части дуги одна и та же, и её сечение — самая
    плотная группа проб.
    """
    values = sorted(float(v) for v in lengths)
    best, count = values[-1], -1
    for value in values:
        near = sum(abs(other - value) <= value * FLANK_BIN for other in values)
        if near > count or (near == count and value > best):
            best, count = value, near
    return best


def _flank_point(product: Product, decor_box=None, want: float = 0.0,
                 avoid=None) -> tuple[float, float]:
    """Точка промера ширины шинки: где кадр показывает лицевую грань целиком.

    Широта центра просвета для этого не годится, хотя шинка там и стоит
    вертикально: на боку кольца камера видит её с торца, сечение там меряет
    толщину, и метка ширины вставала на тот же размер, что и метка толщины —
    отсюда и жалоба, что две метки шинки показывают одно и то же.

    Из сечений вдоль шинки выбирается то, что сходится с табличной шириной в
    масштабе кадра (`want`): это место и есть лицевая грань. Разброс допускается
    ракурсный — от 0,6 до 1,6 таблицы; вне его сечение принадлежит не грани.
    На выборке из 136 позиций выбор по таблице сходится с ней в 89 % кадров и
    садится на толщину в 10 % — против 52 % и 17 % у выбора по одной геометрии.

    Когда ни одно сечение к таблице не подходит, спорить не с чем: масштаб кадра
    выведен из рамки декоративного элемента, и раздутая рамка портит его вдвое.
    Тогда место выбирает сама геометрия — по самым длинным сечениям, где шинка
    повёрнута к камере плоскостью. Берётся не максимум, а верхний перцентиль:
    максимум сидит на стыке с кастом, где к шинке прибавляется плечо.

    Длину интервала это правило не трогает: она по-прежнему из таблицы. По кадру
    выбирается только место, и потому уточнение рамки его не раскачивает — сдвиг
    масштаба переставляет метку на соседнюю пробу того же участка грани.

    Годных проб может не остаться вовсе — тогда возвращается None, и ширина уходит
    на выноску со стрелкой. Нарисовать интервал наугад хуже: у кольца с жемчугом
    контур просвета упирается в саму жемчужину, и метка садилась на неё.
    """
    samples, trusted = _flank_candidates(product, decor_box)
    if samples:
        # таблица говорит, сколько места метке нужно: сечение короче табличной
        # ширины интервал не вместит, и он выйдет короче собственного числа.
        # Такие пробы отбрасываем — но только по чистой выборке: при раздутой рамке
        # `want` вдвое больше правды, и по нему не остаётся ничего, кроме каста
        fit = [s for s in samples if s[0] >= want] if want > 0 and trusted else []
        pool = fit or samples
        # из вместительных проб берём ровное место — самое населённое сечение:
        # шинка на большей части дуги одна и та же, а редкие пробы сидят там, где
        # она уходит под каст или показана в торец
        target = _typical([length for length, _ in pool])
        window = [s for s in pool if abs(s[0] - target) <= target * FLANK_BIN]
        if not window:
            window = [min(pool, key=lambda s: abs(s[0] - target))]
        # метка шинки не должна налезать на метку толщины: обе читаются парой,
        # а сойдясь в одном месте — сливаются в кашу
        if avoid:
            apart = BAND_APART * product.size
            far = [s for s in window if math.dist(s[1], avoid) >= apart]
            if far:
                window = far
        # из оставшегося берём место, где кольцо показывает шинку плашмя
        return max(window, key=lambda s: _face_on(product, s[1]))[1]
    return None


def _face_on(product: Product, point) -> float:
    """Насколько хорошо кадр показывает в этой точке ширину шинки, 0..1.

    Ширина шинки идёт вдоль оси кольца, а та на кадре смотрит вдоль малой оси
    эллипса силуэта. У концов большой оси — по бокам кольца — ось совпадает с
    ходом самой шинки, ширина уходит в торец, и сечение поперёк хода меряет там
    что угодно, кроме неё. У концов малой оси — сверху и снизу — ширина лежит
    поперёк хода целиком, и мерить её надо там.

    Раньше на этом месте стояло «чем ниже по дуге, тем лучше». Правило совпадало
    с истиной только на кольце, повёрнутом ровно: у наклонённого низ силуэта и
    конец малой оси — разные места, и метка садилась на торец.

    К оценке добавлен небольшой перевес нижней половине: сверху на кольце сидит
    декоративный элемент, и шинка там уходит под каст.
    """
    shape = scale.ellipse(product)
    if shape is None:
        return float(point[1])
    vx, vy = point[0] - shape.center[0], point[1] - shape.center[1]
    norm = math.hypot(vx, vy)
    if norm < 1e-6:
        return 0.0
    along = abs((vx * shape.uminor[0] + vy * shape.uminor[1]) / norm)
    return along + (0.15 if point[1] >= shape.center[1] else 0.0)


def _band_mask(product: Product) -> np.ndarray:
    """Металл изделия без бликовых дыр, но с просветом кольца.

    Тонкая маска считает металлом всё, что темнее белого, — и потому дырявится
    на бликах: полированная шинка под прямым светом уходит в чистый белый, и луч,
    идущий поперёк сечения, обрывается на блике в пяти пикселях от кромки. На
    широком ободке так терялось девятнадцать проб из двадцати одной.

    Залитый силуэт этой беды не знает, но у него нет просвета: сечение по нему
    пошло бы через всё кольцо насквозь. Поэтому берём силуэт и вырезаем из него
    просвет — единственную дыру, которая на кадре настоящая.
    """
    cached = product.__dict__.get("_band_metal")
    if cached is not None:
        return cached
    mask = product.solid.copy()
    hole = _hole(product)
    if hole is not None:
        mask = mask & ~hole
    product.__dict__["_band_metal"] = mask
    return mask


def _outer_run(product: Product, start, direction, limit: float | None = None):
    """Наружная кромка изделия на луче и непрерывный пробег металла за ней.

    Луч идёт из центра кольца наружу, поэтому первым найденным металлом всегда
    оказывается тот, что лежит на самом краю силуэта, — а не деталь, попавшая
    между центром и краем. Возвращает (точка на наружной кромке, точка на
    внутренней кромке этого же куска металла).
    """
    dx, dy = direction
    norm = math.hypot(dx, dy) or 1.0
    dx, dy = dx / norm, dy / norm
    limit = int(limit if limit is not None else product.size)
    metal = _band_mask(product)
    far = None
    for step in range(limit):
        x, y = int(round(start[0] + dx * step)), int(round(start[1] + dy * step))
        if not (0 <= x < product.size and 0 <= y < product.size):
            break
        if metal[y, x]:
            far = step
    if far is None:
        return None
    inner = far
    while inner > 0:
        x, y = int(round(start[0] + dx * (inner - 1))), int(round(start[1] + dy * (inner - 1)))
        if not (0 <= x < product.size and 0 <= y < product.size) or not metal[y, x]:
            break
        inner -= 1
    if far - inner < 1:
        return None
    return ((start[0] + dx * far, start[1] + dy * far),
            (start[0] + dx * inner, start[1] + dy * inner))


def _band_probes(product: Product, row, frame: scale.Scale, avoid=None,
                 decor_box=None) -> list:
    """Сечения шинки там, где кадр показывает её ширину, по убыванию правдоподобия.

    Прежний промер шёл по контуру просвета: от каждой его точки луч наружу до
    первого металла, и сечение поперёк хода шинки в этой точке. У ровно снятого
    кольца это работало, у наклонённого — нет. Просвет наклонённого кольца смещён
    к ближнему краю, и луч из его контура наружу попадает не на лицевую грань, а
    на внутреннюю поверхность дальней стенки: метка садилась внутрь кольца и
    меряла ширину, которой там нет.

    Здесь промер привязан к малой оси силуэта — направлению, в котором на кадр
    проецируется ось кольца. На её концах шинка повёрнута к камере плоскостью, её
    ширина лежит поперёк хода и видна целиком; на концах большой оси она уходит в
    торец. Луч идёт из центра наружу, кромку берёт крайнюю, а сечение —
    непрерывный пробег металла внутрь от неё.

    Лучей пускается веер в обе стороны от малой оси: один луч на ажурном кольце
    проваливается в просечку, а по вееру видно, какой пробег на этой дуге
    типичен. Годным считается тот, что сошёлся с ожиданием — толщиной,
    сокращённой на косинус наклона, плюс шириной, сокращённой на синус. Луч,
    попавший на декоративный элемент вместо шинки, выдаёт себя сразу: у кольца с
    ореолом верхний конец даёт впятеро больше ожидаемого.

    Возвращается несколько мест, разведённых по дуге. Второе нужно
    размеру-двойнику: у кольца, декор которого идёт по всей шинке, ширина
    элемента и ширина шинки — один и тот же миллиметр, и две метки на одном
    месте слились бы.
    """
    shape = scale.ellipse(product)
    if shape is None:
        return []
    want = _band_section_want(frame, row)
    ux, uy = shape.uminor
    samples = []
    for sign in (1, -1):
        for k in range(-PROBE_FAN, PROBE_FAN + 1):
            angle = math.radians(k * PROBE_STEP)
            cos, sin = math.cos(angle), math.sin(angle)
            direction = (sign * (ux * cos - uy * sin), sign * (ux * sin + uy * cos))
            run = _outer_run(product, shape.center, direction)
            if run is None:
                continue
            length = math.dist(*run)
            if length >= PROBE_MIN:
                samples.append((length, run, math.atan2(direction[1], direction[0])))
    if not samples:
        return []
    if want <= 0:
        # сверить не с чем — берём место, где металла больше: на противоположной
        # дуге шинка либо ушла под декор, либо закрыта просветом
        good = sorted(samples, key=lambda s: -s[0])
    else:
        good = sorted((s for s in samples
                       if abs(math.log(s[0] / want)) <= math.log(PROBE_ALARM)),
                      key=lambda s: abs(math.log(s[0] / want)))
    if not good:
        return []

    picked: list = []
    for length, run, angle in good:
        middle = _middle(run)
        if avoid and math.dist(middle, avoid) < BAND_APART * product.size:
            continue
        if any(_arc_gap(angle, other) < math.radians(PROBE_APART) for _, other in picked):
            continue
        picked.append((run, angle))
        if len(picked) >= 2:
            break
    if not picked:
        picked = [(good[0][1], good[0][2])]
    return [run for run, _ in picked]


def _arc_gap(one: float, two: float) -> float:
    """Угол между двумя направлениями, 0..pi."""
    return abs((one - two + math.pi) % (2 * math.pi) - math.pi)


def _reads_thickness(measured: float, frame: scale.Scale, values: dict) -> bool:
    """Промер попал в толщину шинки вместо ширины — значит, ширины на кадре нет.

    Кольцо, снятое в фас, показывает торец шинки: её ширина идёт прямо в камеру и
    на кадре не лежит нигде. Любое сечение поперёк ободка меряет там толщину. Само
    по себе сечение об этом не говорит — говорит таблица: если промер сошёлся с
    толщиной, а ширина по таблице заметно больше, показывать этот отрезок как
    ширину нельзя, он чужой.

    Проверка идёт по таблице и масштабу, а не по наклону: наклон считается по
    силуэту, а силуэт у кольца с высоким кастом — не кольцо, и ракурс по нему
    выходит вдвое круче настоящего.
    """
    thickness = frame.px(float(values.get("band_t") or 0))
    width = frame.px(float(values.get("band_w") or 0))
    if thickness < 4 or width <= thickness * 1.4:
        return False
    low, high = BAND_AS_THICKNESS
    return thickness * low <= measured <= thickness * high


def _middle(section) -> tuple:
    """Середина отрезка."""
    (x0, y0), (x1, y1) = section
    return (x0 + x1) / 2, (y0 + y1) / 2


def _band_section_want(frame: scale.Scale, row) -> float:
    """Какой длины ждать сечение шинки на конце малой оси, px.

    Радиус кольца там идёт вглубь кадра под наклоном, ось — тоже, поэтому в один
    отрезок маски укладываются обе величины: толщина, сокращённая на косинус, и
    ширина, сокращённая на синус.
    """
    thickness = float(getattr(row, "band_t", 0) or 0)
    width = float(getattr(row, "band_w", 0) or 0)
    if not frame.ok or (thickness <= 0 and width <= 0):
        return 0.0
    return frame.px(thickness * frame.cos_tilt + width * frame.sin_tilt)


def _flank_fallback(product: Product) -> tuple[float, float]:
    """Куда показать стрелкой, когда промерить ширину на кадре не вышло."""
    px0, py0, px1, py1 = product.bbox
    column = int(px1 - BAND_FLANK * (px1 - px0))
    run = _bottom_run(product, column)
    return float(column), float((run[0] + run[1]) / 2 if run else (py0 + py1) / 2)


def _leader_start(span, elbow) -> tuple[float, float]:
    """Откуда вести полку: ближний к излому конец интервала либо его середина."""
    (x0, y0), (x1, y1) = span
    middle = ((x0 + x1) / 2, (y0 + y1) / 2)
    return min((span[0], span[1], middle),
               key=lambda p: math.dist(p, elbow))


def _mm_per_px(plan: Plan, values: dict, element) -> float:
    """Масштаб кадра по рамке декоративного элемента, мм на пиксель.

    Оставлено разбору позиции (`markup/inspect.py`) и редактору: им нужна именно
    эта мера — та, что зависит от рамки, — чтобы человек видел, куда рамка увела
    масштаб. Раскладка берёт сводный масштаб из `markup/scale.py`.
    """
    return scale.by_decor(plan, values, element)[0]


DECOR_ANCHOR_DEPTH = 0.18   # насколько ниже верхней кромки каста ставится якорь выноски


def _decor_anchor(product: Product, decor: tuple) -> tuple[float, float]:
    """Точка на правой кромке декоративного элемента, у его верхнего угла.

    Рамка от модели обычно шире самой вставки и снизу захватывает шинку, поэтому
    сначала уточняем верх каста по фактическому металлу, а потом берём крайний
    правый пиксель на этой высоте, не выходя за колонки рамки.
    """
    dx0, dy0, dx1, dy1 = (int(v) for v in decor)
    vertical = product.column_extent(dx0, dx1)
    top = max(dy0, vertical[0]) if vertical else dy0
    bottom = min(dy1, vertical[1]) if vertical else dy1
    if bottom - top < 8:
        top, bottom = dy0, dy1

    y = top + (bottom - top) * DECOR_ANCHOR_DEPTH
    band = max(4, int((bottom - top) * 0.04))
    span = product.row_extent(int(y - band), int(y + band), dx0, dx1)
    return (span[1] if span else dx1), y



def _locked_decor(plan: Plan, raw: tuple) -> decor.Decor | None:
    """Декоративный элемент, закреплённый вручную в веб-редакторе.

    Человек, поправивший рамку глазами, видит кадр лучше любой эвристики,
    поэтому закреплённый элемент не проходит ни поиск по контрасту, ни
    выправление по пропорциям — он берётся ровно таким, как задан.
    """
    if plan.decor_lock == "box":
        return decor.from_box(raw)
    if plan.decor_lock == "oriented" and plan.decor_rect:
        rect = plan.decor_rect
        angle = math.radians(float(rect.get("angle", 0.0)))
        axis = (math.cos(angle), math.sin(angle))
        long, short = float(rect["long"]), float(rect["short"])
        center = (float(rect["center"][0]), float(rect["center"][1]))
        half = [(axis[0] * long / 2, axis[1] * long / 2),
                (-axis[1] * short / 2, axis[0] * short / 2)]
        corners = [(center[0] + sx * half[0][0] + sy * half[1][0],
                    center[1] + sx * half[0][1] + sy * half[1][1])
                   for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        return decor.Decor(box=(min(xs), min(ys), max(xs), max(ys)), center=center,
                           axis=axis, long=long, short=short, source="manual",
                           angle=math.degrees(angle))
    return None


def _table_axes(row) -> tuple[float, float]:
    """Табличные «длина x ширина» декоративного элемента, годные для кадра.

    Размер кольца габаритом элемента не является: 54 мм в графе «длина» — это
    обхват пальца, и ни рамку по нему не проверить, ни масштаб из него не
    вывести. Такую длину отдаём нулём — дальше по коду это значит «мерить не по
    чему», и рамка проверяется по одной ширине.
    """
    length = 0.0 if scale.ring_size_mm(row) else float(row.length or 0.0)
    return length, float(row.width or 0.0)


def _decor_miss(product: Product, plan: Plan, row, element) -> float:
    """Насколько рамка расходится с кадром: во сколько раз, чем больше — тем хуже.

    Складывается из двух независимых промахов. Первый — внутренний: две стороны
    рамки дают два масштаба, и разойтись они могут только если рамка меряет не
    то, что записано в таблице. Второй — внешний: масштаб рамки против масштаба
    по кольцу, который о рамке ничего не знает.

    Мера безразмерная, поэтому по ней сравниваются между собой рамки любого
    происхождения — закреплённая человеком, найденная по контрасту, присланная
    моделью.
    """
    value, inner = scale.by_decor(plan, scale.table_dims(row), element)
    if value <= 0:
        return float("inf")
    ring = scale.by_ring(product, plan, row)
    outer = max(value / ring, ring / value) if ring > 0 else 1.0
    return max(inner, outer)


def find_decor(product: Product, plan: Plan, row) -> tuple[decor.Decor, list[str]]:
    """Итоговый декоративный элемент и замечания к нему.

    Кандидатов трое: рамка, закреплённая человеком в редакторе; рамка, найденная
    по контрасту кадра (она приходит вместе со своим наклоном, и вытянутый декор
    по диагонали меряется вдоль собственной оси); рамка модели, поджатая по маске
    к плотной части изделия.

    Выбор между ними идёт по одной мере — расхождению с масштабом кадра
    (`_decor_miss`). Раньше выбор был жёстким по старшинству: закреплённая рамка
    бралась, не глядя. Но закрепляются не только правки, сделанные глазами на
    этом кадре: та же правка переносится на полсотни похожих (`markup/spread.py`),
    и на кадре, который лишь похож, она садится мимо — центром выше изделия,
    стороной в пол-кадра. Такая рамка тянет за собой всё: оба габарита, масштаб,
    место промера шинки.

    Поэтому человек по-прежнему главный, но не безусловно: его рамка уступает,
    только если другая сходится с кадром заметно лучше (`DECOR_PICK`), и об
    этом пишется замечание — приёмке видно, что рамку отменил не случай.
    """
    notes: list[str] = []
    raw = _clip_box(plan.decor_box, product.bbox)
    length_mm, width_mm = _table_axes(row)

    candidates: list[tuple[str, decor.Decor]] = []
    locked = _locked_decor(plan, raw)
    if locked is not None:
        locked = _fit_to_product(product, locked)
    if locked is not None:
        candidates.append(("ручная", locked))

    found = decor.detect(product, raw, length_mm, width_mm,
                         _bore_scale(product, plan, row))
    if found is not None and found.inclined:
        candidates.append(("по контрасту", found))

    # Габарит рисуется по рамке, поэтому её ещё надо поджать к плотной части
    # изделия — и рамку модели, и найденную по контрасту. Обе прихватывают
    # крапаны, плечики и край шинки: профиль маски там уже пологий, а выносная
    # линия садится заметно ниже вставки, и габарит показывает больше, чем есть.
    # Наклонный элемент меряется вдоль собственных осей, его рамка в отрисовке
    # не участвует, и трогать её незачем.
    box = product.dense_box(found.box if found is not None else raw)
    candidates.append(("по кадру", decor.from_box(box)))

    name, element = _pick_decor(product, plan, row, candidates, notes)

    # Пропорции рамки только проверяются, но не правятся. Подгонять рамку под
    # табличное отношение сторон казалось безобидным — режется же пологий край, —
    # а на деле это подгонка под цифру, а не под изделие: у кольца с дорожкой
    # камней вдоль шинки табличная длина измерена по дуге, отношение выходит 4,5
    # против фактических 1,7, и рамка ужималась с 574 пикселей высоты до 240,
    # теряя две трети декора. По контрольной сверке масштаба через просвет кольца
    # подгонка и в среднем вредила: 64 % правдоподобных против 67 % без неё.
    if name == "по кадру":
        h_mm = row.dims().get(plan.axes.get("horizontal"), 0.0)
        v_mm = row.dims().get(plan.axes.get("vertical"), 0.0)
        width, height = box[2] - box[0], box[3] - box[1]
        if h_mm > 0 and v_mm > 0 and min(width, height) >= 4 and not scale.ring_size_mm(row):
            score = abs(math.log((width / height) / (h_mm / v_mm)))
            if score > math.log(1.5):
                notes.append("пропорции рамки расходятся с табличными размерами — нужна проверка")
    return element, notes


def _pick_decor(product: Product, plan: Plan, row, candidates: list, notes: list):
    """Выбирает рамку из кандидатов по расхождению с масштабом кадра."""
    scored = [(name, el, _decor_miss(product, plan, row, el)) for name, el in candidates]
    manual = next((s for s in scored if s[0] == "ручная"), None)
    best = min(scored, key=lambda s: s[2])
    if manual is None:
        return best[0], best[1]
    if best[0] == "ручная" or not (best[2] * DECOR_PICK < manual[2]):
        return manual[0], manual[1]
    notes.append(
        f"рамка, закреплённая вручную, расходится с масштабом кадра в {manual[2]:.1f} раза — "
        f"взята рамка {best[0]}")
    return best[0], best[1]


def _fit_to_product(product: Product, element: decor.Decor) -> decor.Decor | None:
    """Проверяет закреплённую рамку на попадание в изделие.

    Рамку, перенесённую с похожего кадра, случается вынести за силуэт целиком или
    почти целиком: центр оказывается над изделием, в белом поле. Мерить по ней
    нечего — ни габарит, ни масштаб. Такую рамку отбрасываем совсем; ту, что
    большей частью на изделии, оставляем как есть: подрезать повёрнутый
    прямоугольник по осям кадра значит испортить его собственные оси.
    """
    x0, y0, x1, y1 = (int(round(v)) for v in element.box)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(product.size, x1), min(product.size, y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    inside = product.thin[y0:y1, x0:x1]
    if not inside.any():
        return None
    # металл обязан занимать в рамке хоть сколько-то места: рамка, зацепившая
    # изделие уголком, меряет уголок, а не декоративный элемент
    cx, cy = element.center
    if not (0 <= int(cx) < product.size and 0 <= int(cy) < product.size):
        return None
    if inside.sum() < 0.04 * max(1, (x1 - x0) * (y1 - y0)):
        return None
    return element


# ---------------------------------------------------------------- сборка листа

def leader_anchor(product: Product, plan: Plan, element, elbow,
                  cache=None) -> tuple[tuple[float, float], str]:
    """Куда поставить стрелку выноски высоты и откуда это взято.

    Три источника, в порядке убывания веса:

    1. отметка человека на этом кадре — она всегда права;
    2. медиана отметок по такой же паре «тип изделия + ракурс» — привычка
       технолога, накопленная в `cache/якоря.json`;
    3. умолчание: ближайшая к полке точка.

    В любом случае результат притягивается к фактическому металлу внутри
    декоративного элемента. Раньше якорь брался с угла описанной рамки, и на
    наклонном или вытянутом декоре стрелка втыкалась в пустоту либо уезжала на
    шинку — обе жалобы снимаются именно этим притягиванием.
    """
    if plan.leader_lock and plan.leader_anchor:
        точка = (float(plan.leader_anchor[0]), float(plan.leader_anchor[1]))
        снято = product.nearest_in_rect(точка, element.center, element.axis,
                                        element.long, element.short)
        return (снято or точка), "отметка человека"

    цель, источник = elbow, "ближайшая к полке точка"
    if cache is not None:
        try:
            from .anchors import learned, to_frame
            доли = learned(cache, plan.item_type, plan.view)
        except Exception:                      # noqa: BLE001 — журнал необязателен
            доли = None
        if доли:
            цель = to_frame(доли, element)
            источник = "медиана отметок"

    снято = product.nearest_in_rect(цель, element.center, element.axis,
                                    element.long, element.short)
    if снято:
        return снято, источник
    # металла в рамке не нашлось — держимся прежнего угла, но хотя бы на изделии
    запасной = _decor_anchor(product, tuple(element.box))
    return запасной, "угол рамки"


def _inclined_pair(product: Product, element, values: dict, levels: dict,
                   long_key: str, short_key: str) -> list[tuple[str, InclinedDim]]:
    """Два габарита вдоль собственных осей вытянутого наклонного элемента.

    Возвращает их вместе с размером, за который каждый отвечает: дальше габарит
    ещё проверяется на сходимость со своим числом и может уйти на выноску.
    """
    st = STYLE
    cx, cy = element.center
    ux, uy = element.axis
    nx, ny = element.normal
    half_long, half_short = element.long / 2, element.short / 2
    heart = ((product.bbox[0] + product.bbox[2]) / 2, (product.bbox[1] + product.bbox[3]) / 2)

    def away(vx, vy, distance) -> int:
        """Знак, при котором размерная линия уходит от изделия, а не внутрь него."""
        plus = math.dist((cx + vx * distance, cy + vy * distance), heart)
        minus = math.dist((cx - vx * distance, cy - vy * distance), heart)
        return 1 if plus >= minus else -1

    dims = []
    for key, (vx, vy), half, (wx, wy), extent in (
            (long_key, (nx, ny), half_short, (ux, uy), half_long),
            (short_key, (ux, uy), half_long, (nx, ny), half_short)):
        sign = away(vx, vy, half)
        nvx, nvy = vx * sign, vy * sign
        # Концы измеряемого отрезка берём на средней линии элемента, а не на углах
        # описанного прямоугольника: у наклонного элемента углы приходятся на пустое
        # поле, и выносная линия начиналась бы в воздухе. Кромку под каждым концом
        # отрисовка найдёт лучом по нормали.
        a = (cx - wx * extent, cy - wy * extent)
        b = (cx + wx * extent, cy + wy * extent)
        # размерную линию отодвигаем за силуэт: по наклонной оси заранее не известно,
        # где кончается изделие, поэтому расстояние промеряется лучом по маске
        clear = max(product.reach(a, (nvx, nvy)), product.reach(b, (nvx, nvy)),
                    product.reach((cx, cy), (nvx, nvy)))
        offset = clear + st.offset_1 + levels.get(key, 0) * st.offset_step
        dims.append((key, InclinedDim(a=a, b=b, normal=(nvx, nvy), offset=offset,
                                      text=fmt(values[key]), overshoot=2 * half)))
    return dims


def _axial_span(element, a: float, b: float, axis: str) -> tuple[float, float]:
    """Ужимает габарит повёрнутого элемента до его собственного размера вдоль оси.

    Габарит по осям кадра меряет описанный прямоугольник, а он у повёрнутого
    элемента больше самого элемента: у квадратной вставки под 32° — на треть.
    Число рядом остаётся табличным, и линия перестаёт с ним сходиться — приёмка
    видит, что габарит «поехал», а конец его и правда уезжает на плечо шинки.

    Вытянутый элемент это лечит наклонными линиями (`_inclined_pair`); для
    компактного они не годятся — две наклонные над мелким камнем читаются как
    перечёркнутый кадр. Здесь линия остаётся осевой, но берёт длину с собственной
    оси элемента и садится по его центру.

    Работает только там, где собственные оси известны: у рамки от модели их нет,
    и ужимать нечем.
    """
    if element.inclined or element.source not in ("texture", "manual"):
        return a, b
    if element.short <= 0 or abs(element.angle) < decor.INCLINE_MIN_ANGLE:
        return a, b
    ux, uy = element.axis
    along_frame = abs(ux) >= abs(uy)          # длинная ось ближе к горизонтали кадра
    own = element.long if (axis == "h") == along_frame else element.short
    if own >= abs(b - a):
        return a, b
    center = element.center[0] if axis == "h" else element.center[1]
    return center - own / 2, center + own / 2


def _runs(line) -> list[tuple[int, int]]:
    """Непрерывные пробеги металла в срезе маски."""
    idx = np.where(line)[0]
    if not len(idx):
        return []
    breaks = np.where(np.diff(idx) > 1)[0]
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [len(idx) - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def _tail(product: Product, edge: float, band: tuple, axis: str,
          direction: int, limit: float) -> float:
    """Продолжает габарит за кромку рамки, пока идёт тот же кусок металла.

    Рамка декоративного элемента ставится по его плотной части, и сходящий на нет
    хвост в неё не попадает: у кольца с дорожкой паве вдоль шинки дорожка сужается
    и мельчает, рамка обрывается на полпути, а габарит показывает меньше, чем есть.

    Поэтому от кромки рамки идёт марш наружу: пробег металла обязан перекрываться с
    предыдущим и не толстеть — ни от шага к шагу, ни против кромки рамки.

    Но одного сужения мало: плечо шинки, уходящее из полосы, тоже сужается и тоже
    кончается на контуре. Поэтому от хвоста требуется ещё и быть хвостом: на кромке
    рамки он уже тонкий — не толще трети поперечника рамки. Дорожка паве к своему
    концу истончается до нитки и это условие держит; плечо шинки подходит к рамке во
    всю свою толщину и отсеивается сразу.

    И марш засчитывается, только если упёрся в край самого изделия — в срезе не
    осталось металла нигде, не только в полосе. Дойти до предела (доля стороны рамки)
    — отказ: настолько длинный хвост уже не хвост. При отказе габарит остаётся на
    кромке рамки, то есть правило умеет только удлинить габарит до настоящего конца
    элемента и никогда — увести его на чужую деталь.
    """
    lo, hi = int(max(0, band[0])), int(min(product.size, band[1]))
    if hi - lo < 2 or not direction:
        return edge
    step = 1 if direction > 0 else -1

    def section(k: int) -> list[tuple[int, int]]:
        if not (0 <= k < product.size):
            return []
        return _runs(product.thin[lo:hi, k] if axis == "h" else product.thin[k, lo:hi])

    pos = int(round(edge))
    runs = section(pos)
    if not runs:
        return edge
    current = max(runs, key=lambda r: r[1] - r[0])      # на кромке элемент — главный металл
    thick = current[1] - current[0]
    if thick > TAIL_THIN * (hi - lo):
        return edge
    out = edge
    for _ in range(int(limit)):
        pos += step
        follow = None
        for run in section(pos):
            if run[0] - TAIL_JOIN <= current[1] and run[1] + TAIL_JOIN >= current[0]:
                follow = run if follow is None else (min(follow[0], run[0]), max(follow[1], run[1]))
        if follow is None:
            whole = product.thin[:, pos] if axis == "h" else product.thin[pos]
            return out if not whole.any() else edge
        width = follow[1] - follow[0]
        if width > min(current[1] - current[0] + TAIL_SLACK, thick):
            return edge
        current, out = follow, float(pos)
    return edge


def _fits(span_px: float, mm: float, frame: scale.Scale) -> bool:
    """Сходится ли габарит со своим числом в масштабе кадра.

    Габарит — это утверждение о кадре: «вот отсюда досюда, и это столько-то
    миллиметров». Утверждение проверяемое, и проверять его обязательно: линия во
    всю ширину кольца, подписанная «3 мм», хуже отсутствия линии — она не
    неточная, она неверная. По контрольной выборке из 178 кадров у 81 два
    габарита противоречили друг другу по масштабу: один и тот же кадр говорил,
    что в миллиметре то 20 пикселей, то 250.
    """
    if not frame.ok or mm <= 0 or span_px < 4:
        return False
    want = frame.px(mm)
    if want < 4:
        return False
    return 1 / GABARIT_TOL <= span_px / want <= GABARIT_TOL


def _band_want(frame: scale.Scale, values: dict) -> float:
    """Сколько пикселей займёт ширина шинки на этом кадре — с поправкой на ракурс.

    Ширина шинки идёт вдоль оси кольца, то есть вглубь кадра. Плашмя лежащее
    кольцо не показывает её вовсе, поставленное на ребро — показывает целиком, а
    на «трёх четвертях» она сокращается на синус наклона. Табличная ширина,
    переведённая в пиксели напрямую, поэтому всегда завышена, и на сильно
    наклонённом кольце — вдвое: ни одна проба сечения под неё не подходила, и
    выбор места промера сваливался на запасную ветку.
    """
    want = frame.px(values.get("band_w", 0.0))
    return want * frame.sin_tilt if want else 0.0


def _demoted_leader(product: Product, element, key: str, values: dict, elbow,
                    st=STYLE) -> Leader:
    """Выноска вместо габарита: размер есть в таблице, но кадром не показан.

    Тот же приём, которым показывается высота декоративного элемента, — и по той
    же причине. Размер кольца на кадре не лежит нигде: 54 мм — это длина дуги, а
    не отрезка. Ширина шинки, записанная в графу декоративного элемента, лежит
    поперёк ободка, а не по краям описанной рамки. Рисовать под такое число
    габарит — значит показывать пальцем не туда.
    """
    return Leader(
        anchor=element.center, elbow=elbow, direction=-1, length=st.leader_len,
        text=fmt(values[key]), caption=CAPTIONS[key],
        # якорь — центр элемента, но конец выноски садится на силуэт: луч идёт от
        # излома к центру и останавливается на первом металле
        snap=True)


def _поправки(plan: Plan, product: Product, cache) -> dict:
    """Что накладывается поверх расчёта: привычки, а сверху — правки этого кадра.

    Привычка — это медиана по десятку кадров, правка — прямое указание человека
    здесь и сейчас. Поэтому они не спорят, а складываются по полям: выученный
    вынос размерной линии остаётся в силе на тех размерах, которых человек на
    этом кадре не касался.
    """
    from . import habits

    итог = {}
    try:
        итог = habits.resolve(cache, plan, product)
    except Exception:                       # noqa: BLE001 — журнал необязателен
        итог = {}
    for ключ, поправка in (plan.tweaks or {}).items():
        итог[ключ] = {**итог.get(ключ, {}), **поправка}
    return итог


def compose(product: Product, plan: Plan, row, levels: dict | None = None,
            element=None, cache=None) -> Sheet:
    """Строит лист разметки. `levels` задаёт номер эшелона для каждого размера."""
    st = STYLE
    levels = levels or {}
    px0, py0, px1, py1 = product.bbox
    element = element if element is not None else find_decor(product, plan, row)[0]
    dx0, dy0, dx1, dy1 = element.box
    # Ручные правки и выученные привычки кладутся в лист одним словарём: раскладка
    # накладывает их в `Sheet.add`, куда бы размер ни попал — на габарит или на
    # выноску. Правка этого кадра главнее привычки: здесь человек сказал больше.
    sheet = Sheet(product=product, tweaks=_поправки(plan, product, cache))
    values = row.dims()

    def закреплён(key: str) -> bool:
        """Концы габарита проставлены руками — размер остаётся габаритом.

        Иначе правка обернулась бы против себя: человек дотянул габарит до конца
        элемента, длина разошлась с табличным числом сильнее прежнего, и размер
        уехал бы на выноску — туда, где эту правку уже не видно.
        """
        поправка = sheet.tweaks.get(key) or {}
        return "a" in поправка or "b" in поправка

    # масштаб кадра нужен раньше первой линии: по нему решается, какой размер
    # вообще показывать габаритом
    frame = scale.measure(product, plan, row, element)
    note = scale.bore_check(product, frame, plan)
    if note:
        sheet.notes.append(note)
    if frame.source == "кольцо" and frame.by_decor > 0:
        sheet.notes.append(
            "масштаб взят по кольцу: рамка декоративного элемента расходится с ним "
            f"в {frame.spread:.1f} раза")

    def level_of(key: str, base: float, direction: int) -> float:
        return base + direction * (st.offset_1 + levels.get(key, 0) * st.offset_step)

    #: размеры, которые кадр показать не смог, — уходят на выноски слева
    demoted: list[str] = []
    #: размер декоративного элемента, совпавший с размером шинки, габаритом не
    #: показывается вовсе: он лежит поперёк ободка, а не по краям рамки
    twins = scale.band_twins(row)

    if element.inclined:
        # 1–2. вытянутый декор лежит по диагонали — меряем его вдоль собственных осей
        long_key, short_key = (("length", "width") if values["length"] >= values["width"]
                               else ("width", "length"))
        for key, dim in _inclined_pair(product, element, values, levels,
                                       long_key, short_key):
            if key not in twins and (_fits(math.dist(dim.a, dim.b), values[key], frame)
                                     or закреплён(key)):
                sheet.add(dim, key)
            else:
                demoted.append(key)
    else:
        # 1. горизонтальный габарит декоративного элемента — сверху.
        # Концы габарита берём не по рамке, а по фактическому металлу: рамка модели
        # регулярно чуть шире вставки, и габарит заезжает на плечо шинки. Внутри
        # рамки металл только сужает габарит, а хвост элемента, в рамку не попавший,
        # возвращает марш `_tail`.
        h_key = plan.axes["horizontal"]
        span = product.row_extent(int(dy0), int(dy1), int(dx0), int(dx1))
        a, b = span if span else (dx0, dx1)
        tail = TAIL_LIMIT * (dx1 - dx0)
        a = _tail(product, a, (dy0, dy1), "h", -1, tail)
        b = _tail(product, b, (dy0, dy1), "h", +1, tail)
        a, b = _axial_span(element, a, b, "h")
        extent = product.column_extent(int(a), int(b), int(dy0), int(dy1))
        if h_key not in twins and (_fits(abs(b - a), values[h_key], frame)
                                   or закреплён(h_key)):
            sheet.add(LinearDim(
                axis="h", a=a, b=b,
                base=extent[0] if extent else dy0,
                level=level_of(h_key, py0, -1),
                text=fmt(values[h_key]), side=-1,
                box=element.box,
            ), h_key)
        else:
            demoted.append(h_key)

        # 2. вертикальный габарит декоративного элемента — слева
        v_key = plan.axes["vertical"]
        column = product.column_extent(int(dx0), int(dx1), int(dy0), int(dy1))
        a, b = column if column else (dy0, dy1)
        tail = TAIL_LIMIT * (dy1 - dy0)
        a = _tail(product, a, (dx0, dx1), "v", -1, tail)
        b = _tail(product, b, (dx0, dx1), "v", +1, tail)
        a, b = _axial_span(element, a, b, "v")
        extent = product.row_extent(int(a), int(b), int(dx0), int(dx1))
        if v_key not in twins and (_fits(abs(b - a), values[v_key], frame)
                                   or закреплён(v_key)):
            sheet.add(LinearDim(
                axis="v", a=a, b=b,
                base=extent[0] if extent else dx0,
                level=level_of(v_key, px0, -1),
                text=fmt(values[v_key]), side=-1,
                box=element.box,
            ), v_key)
        else:
            demoted.append(v_key)

    # 3. третий габарит декоративного элемента не виден на ракурсе — выноска вправо-вверх.
    # Якорь сажаем на верхний угол каста: ниже к нему подходит шинка, и стрелка
    # начинает указывать на ободок вместо вставки. Ищем только внутри рамки
    # декоративного элемента и только по фактическому металлу.
    l_key = plan.leader_dim
    elbow = (px1 + st.offset_1 * 0.55, level_of(l_key, py0, -1) * 0.5 + py0 * 0.5)
    anchor, anchor_source = leader_anchor(product, plan, element, elbow, cache)
    sheet.add(Leader(
        anchor=anchor, elbow=elbow,
        direction=+1, length=st.leader_len,
        text=fmt(values[l_key]), caption=CAPTIONS[l_key],
        # якорь уже стоит на металле внутри декоративного элемента, досаживать
        # его лучом нельзя: луч от полки первым делом упрётся в шинку
        snap=False,
    ), l_key)
    sheet.anchor_source = anchor_source

    # 3а. размеры, не сошедшиеся с кадром, — выносками с полкой влево: первая над
    # изделием, вторая под ним.
    #
    # Сбоку их поставить не выходит. Подпись вроде «Ширина декоративного элемента»
    # — это полтысячи пикселей, а поле слева от изделия шириной в четыреста: подпись
    # упирается в край холста, отжимается обратно и ложится на металл. Сверху и
    # снизу поле идёт во всю ширину кадра, и там она встаёт свободно — тем же
    # порядком, что и в эталонах, где длинная подпись высоты стоит над изделием.
    on_leader = [key for key in demoted if key not in twins]
    for order, key in enumerate(on_leader[:2]):
        step = st.offset_1 * (0.6 + levels.get(key, 0) * 0.5)
        elbow_x = px0 + (px1 - px0) * 0.18
        # нижняя выноска уходит ниже метки толщины шинки: обе стоят внизу, и на
        # одной высоте их полки сходятся
        elbow_y = (py0 - step) if order == 0 else (py1 + step + st.offset_1 * 0.75)
        sheet.add(_demoted_leader(product, element, key, values,
                                  (elbow_x, elbow_y), st), key)

    # Размеры шинки: ширина — интервалом прямо на металле (размерная линия поперёк
    # сечения, засечки на кромках, число на полке рядом), толщина — выноской со
    # стрелкой. Стрелка показывает только направление, интервал — что именно
    # померено.

    def in_px(key: str) -> float:
        """Табличный размер в пикселях этого кадра."""
        return frame.px(values[key])

    def checked(measured: float, key: str) -> None:
        """Сверяет сечение с таблицей и пишет замечание при явном промахе масштаба.

        Умеренное расхождение — это ракурс: в сечение шинки попадает и её
        плоскость, и торец. Замечание пишется, когда расхождение уже не
        объяснить ракурсом — там промахнулся масштаб кадра, а не промер.
        """
        want = in_px(key)
        if want and not (want / BAND_ALARM <= measured <= want * BAND_ALARM):
            sheet.notes.append(f"{CAPTIONS[key].lower()}: промер кадра расходится"
                               " с таблицей в разы — проверьте масштаб")

    def band_dim(key: str, point, elbow, direction: int, arrow: str,
                 span_ok: bool = True, section=None, force_span: bool = False) -> None:
        """Размер шинки: интервал по сечению в точке плюс полка с числом.

        Сечение берётся поперёк хода шинки, а не по оси кадра: осевой разрез на
        косом участке растягивается на наклон, и оба размера шинки выходили на
        кадре одинаковой длины.
        """
        section = section or _band_section(product, point)
        want = in_px(key)
        unreadable = 0 < want < MIN_BAND_PX
        allowed = force_span or key in BAND_SPANS
        if not allowed or not section or unreadable or not span_ok:
            edge = product.row_extent(int(point[1] - 6), int(point[1] + 6))
            sheet.add(Leader(
                anchor=(edge[1] if edge and arrow == "h" else point[0], point[1]),
                elbow=elbow, direction=direction, length=st.leader_len,
                text=fmt(values[key]), caption=CAPTIONS[key], arrow=arrow,
                overshoot=st.offset_1), key)
            return
        outer, inner = section
        checked(math.dist(outer, inner), key)
        span = _band_interval(outer, inner, _face_share(frame, values))
        sheet.add(Leader(
            anchor=_leader_start(span, elbow), elbow=elbow, direction=direction,
            length=st.leader_len, text=fmt(values[key]), caption=CAPTIONS[key],
            span=span, snap=False), key)

    # 4. толщина шинки — выноской со стрелкой к нижней дуге кольца.
    #
    # Интервалом на металле её показать нечем. Камера на этом ракурсе видит лицевую
    # грань шинки, а стенку за ней не видит, поэтому любой прямой разрез читает
    # ширину: на 106 позициях выборки, где размеры различаются, сечение показывало
    # вдвое больше табличной толщины. Разделить сечение по яркости — на грань и торец
    # — тоже не выходит: блик на полированном металле гуляет. Отсюда и жалоба
    # «толщина ничем не отличается от ширины»: обе метки кроют одну и ту же грань.
    #
    # Размер, которого на кадре не видно, в эталонах показывают выноской с мелкой
    # подписью — тем же способом, что и высоту декоративного элемента. Так две метки
    # шинки уже не спутать: у ширины интервал с засечками, у толщины стрелка.
    band_col = _band_column(product, plan.band_point)
    run = _bottom_run(product, band_col)
    thickness_at = (band_col, (run[0] + run[1]) / 2 if run else py1)
    band_dim("band_t", thickness_at,
             (band_col, py1 + st.offset_1 * (0.6 + levels.get("band_t", 0) * 0.5)),
             direction=-1, arrow="v")

    # 5. ширина шинки — поперёк лицевой грани, на том участке кольца, где кадр
    # эту грань показывает целиком (`_flank_point`). На боку кольца шинка
    # повёрнута к камере торцом, и интервал там повторял метку толщины.
    # Место промера ширины: конец малой оси кольца, где шинка повёрнута к камере
    # плоскостью. Не вышло (изделие не кольцо, эллипс не читается) — остаётся
    # прежний путь по контуру просвета.
    probes = _band_probes(product, row, frame, thickness_at, element.box)
    section = probes[0] if probes else None
    if section is None:
        flank = _flank_point(product, element.box, _band_want(frame, values), thickness_at)
        at = flank or _flank_fallback(product)
        readable = flank is not None
    else:
        at = _middle(section)
        readable = not _reads_thickness(math.dist(*section), frame, values)
    band_dim("band_w", at,
             (px1 + st.offset_1 * 0.55, at[1] + levels.get("band_w", 0) * st.offset_step),
             direction=+1, arrow="h", span_ok=readable, section=section)

    # 6. размер декоративного элемента, совпавший с шириной шинки, показывается тем
    # же способом — интервалом поперёк ободка, а не стрелкой в пустоту. Это один и
    # тот же миллиметр, и читаться он должен одинаково; чтобы две метки не легли
    # друг на друга, вторая садится на противоположный конец малой оси.
    for key in (k for k in demoted if k in twins):
        twin = probes[1] if len(probes) > 1 and readable else None
        if twin is None:
            # второго годного места на кольце нет — показываем стрелкой на ту же
            # шинку, а не выноской в центр изделия: размер всё-таки её, и палец
            # должен указывать на неё
            step = st.offset_1 * (0.6 + levels.get(key, 0) * 0.5)
            spot = _middle(section) if section else element.center
            sheet.add(Leader(
                anchor=spot, elbow=(px0 + (px1 - px0) * 0.18, py1 + step + st.offset_1 * 0.75),
                direction=-1, length=st.leader_len, text=fmt(values[key]),
                caption=CAPTIONS[key], arrow="v", snap=False), key)
            continue
        spot = _middle(twin)
        # эшелон уводит полку вниз, а не влево: слева от изделия всего четыреста
        # пикселей поля, и длинной подписи там не разойтись — она упирается в край
        # холста и отжимается обратно на металл. Вниз поле открытое
        drop = levels.get(key, 0) * st.offset_step
        band_dim(key, spot, (px0 - st.offset_1 * 0.55, spot[1] + drop),
                 direction=-1, arrow="h", section=twin, force_span=True)

    return sheet


# ---------------------------------------------------------------- разрешение конфликтов

def _overlap(a, b, pad: int = 0) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0]
                or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def _outside_canvas(box, size: int, margin: float) -> bool:
    return box[0] < margin or box[1] < margin or box[2] > size - margin or box[3] > size - margin


def _shrink(box, frac: float = 0.14):
    """Ужимает рамку подписи: касание линии по самому краю глифов не мешает чтению.

    Отступ ограничен сверху. Доля от ширины у длинной подписи вроде «Высота
    декоративного элемента» даёт полсотни пикселей с каждого края, и размерная
    линия, зашедшая под её первые буквы, проверку проходила.
    """
    cap = STYLE.text_gap
    x0, y0, x1, y1 = box
    dx = min((x1 - x0) * frac, (x1 - x0) / 2 - 1, cap)
    dy = min((y1 - y0) * frac, (y1 - y0) / 2 - 1, cap)
    return x0 + dx, y0 + dy, x1 - dx, y1 - dy


def _segment_hits(segment, box) -> bool:
    """Проходит ли отрезок сквозь рамку. Отсечение Лианга — Барски."""
    x0, y0, x1, y1 = segment
    dx, dy = x1 - x0, y1 - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - box[0]), (dx, box[2] - x0),
                 (-dy, y0 - box[1]), (dy, box[3] - y0)):
        if abs(p) < 1e-9:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def solve(product: Product, plan: Plan, row, renderer: Renderer, cache=None,
          levels_out: dict | None = None):
    """Раскладывает разметку, раздвигая эшелоны, пока подписи конфликтуют.

    Возвращает (изображение, список рамок подписей, список замечаний).

    `levels_out` — куда положить эшелоны, на которых раскладка остановилась.
    Редактору они нужны, чтобы показывать на схеме ту же геометрию, что легла в
    PNG: правится мышью именно она, и разъехаться им нельзя.
    """
    element, decor_notes = find_decor(product, plan, row)
    values = row.dims()
    if element.inclined:
        first, second = (("length", "width") if values["length"] >= values["width"]
                         else ("width", "length"))
    else:
        first, second = plan.axes["horizontal"], plan.axes["vertical"]
    levels = {key: 0 for key in (first, second, plan.leader_dim, "band_w", "band_t")}
    keepout = product.keepout(pad=int(STYLE.ext_gap))
    best = None
    best_score = None      # число конфликтов в лучшем проходе

    for _ in range(MAX_PASSES):
        sheet = compose(product, plan, row, levels, element, cache)
        sheet.notes.extend(decor_notes)
        image, boxes, segments = renderer.render(sheet)
        issues: list[str] = []
        guilty: set[str] = set()

        def key_of(index: int) -> str:
            """Размер, за который отвечает элемент разметки под этим номером.

            Спрашивается у самого листа: очередь элементов больше не постоянна —
            размер, не сошедшийся с кадром, уходит с габарита на выноску и встаёт
            в другое место списка.
            """
            return sheet.key_at(index) or first

        for index, box in boxes:
            key = key_of(index)
            if _outside_canvas(box, product.size, STYLE.min_margin):
                issues.append(f"подпись «{CAPTIONS[key]}» выходит за холст")
                guilty.add(key)
            x0, y0, x1, y1 = (int(max(0, box[0])), int(max(0, box[1])),
                              int(min(product.size, box[2])), int(min(product.size, box[3])))
            if x1 > x0 and y1 > y0 and keepout[y0:y1, x0:x1].any():
                issues.append(f"подпись «{CAPTIONS[key]}» накрывает изделие")
                guilty.add(key)

        # линии разметки тоже участвуют в контроле: наклонный участок выноски
        # раньше проходил прямо под своей же мелкой подписью
        for index, box in boxes:
            inner = _shrink(box)
            if any(_segment_hits(segment, inner) for _, segment in segments):
                issues.append(f"линия разметки проходит под подписью «{CAPTIONS[key_of(index)]}»")
                guilty.add(key_of(index))

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if boxes[i][0] == boxes[j][0]:
                    continue
                if _overlap(boxes[i][1], boxes[j][1], pad=int(STYLE.text_gap)):
                    issues.append(
                        f"подписи «{CAPTIONS[key_of(boxes[i][0])]}» и "
                        f"«{CAPTIONS[key_of(boxes[j][0])]}» перекрываются")
                    guilty.add(key_of(boxes[j][0]))

        # сравниваем проходы по конфликтам, а не по длине комментария: замечания
        # раскладки от прохода к проходу не меняются и выбор бы только путали
        if best_score is None or len(issues) < best_score:
            best_score = len(issues)
            best = (image, boxes, issues + sheet.notes)
            if levels_out is not None:
                levels_out.clear()
                levels_out.update(levels)
        if not issues:
            return image, boxes, list(sheet.notes)

        moved = False
        for key in guilty:
            if levels.get(key, 0) < MAX_PASSES - 1:
                levels[key] = levels.get(key, 0) + 1
                moved = True
        if not moved:
            break

    return best
