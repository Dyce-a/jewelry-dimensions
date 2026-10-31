"""Загрузка фото, отделение изделия от белого фона и подготовка холста."""
from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .config import BG, CANVAS, PRODUCT_MAX_FRAC

WHITE_TOL = 8          # насколько пиксель может отличаться от белого и остаться фоном
GROW_LIMIT = 2.5       # во сколько раз рамке позволено вырасти при подгонке пропорций
MIN_VIEW_FRAC = 0.002  # доля площади кадра, ниже которой компонента считается мусором


# ---------------------------------------------------------------- загрузка

class FetchError(RuntimeError):
    pass


def fetch(urls: list[str], cache_dir: Path, key: str, timeout: int = 60,
          attempts: int = 3) -> Path:
    """Скачивает первое доступное фото (1500, при отказе — 1000) с кэшированием.

    Каждая ссылка пробуется несколько раз: отдача картинок регулярно рвётся
    на середине (IncompleteRead), и повтор почти всегда закрывает вопрос.
    """
    import requests

    if not urls:
        raise FetchError("ссылки на фото отсутствуют")

    cache_dir.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for url in urls:
        # Фото, загруженное человеком через редактор, лежит на диске рядом, а не
        # в облаке поставщика. Ссылкой на него служит сам путь: качать нечего.
        свой = Path(url)
        if свой.exists() and свой.is_file():
            return свой
        digest = hashlib.sha1(url.encode()).hexdigest()[:10]
        dest = cache_dir / f"{key}_{digest}.jpg"
        if dest.exists() and dest.stat().st_size > 1024:
            return dest
        for attempt in range(attempts):
            try:
                response = requests.get(url, timeout=timeout)
                response.raise_for_status()
                payload = response.content
                expected = response.headers.get("Content-Length")
                if expected and len(payload) != int(expected):
                    raise FetchError(f"получено {len(payload)} из {expected} байт")
                if len(payload) < 1024:
                    raise FetchError(f"слишком маленький ответ ({len(payload)} байт)")
                dest.write_bytes(payload)
                Image.open(dest).verify()
                return dest
            except Exception as exc:           # noqa: BLE001 — повтор, затем следующая ссылка
                dest.unlink(missing_ok=True)
                if attempt == attempts - 1:
                    errors.append(f"{url}: {exc}")
                else:
                    time.sleep(1.5 * (attempt + 1))
    raise FetchError("не удалось загрузить фото — " + "; ".join(errors))


# ---------------------------------------------------------------- сегментация

def _masks(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Возвращает (тонкая маска металла, залитый силуэт)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    thin = ((255 - gray.astype(np.int16)) > WHITE_TOL).astype(np.uint8)
    thin = cv2.morphologyEx(thin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    thin = cv2.morphologyEx(thin, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # убираем мелкий мусор
    count, labels, stats, _ = cv2.connectedComponentsWithStats(thin, 8)
    keep = np.zeros(count, bool)
    floor = MIN_VIEW_FRAC * thin.size
    for i in range(1, count):
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= floor
    thin = keep[labels].astype(np.uint8)

    contours, _ = cv2.findContours(thin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    solid = np.zeros_like(thin)
    cv2.drawContours(solid, contours, -1, 1, thickness=cv2.FILLED)
    return thin.astype(bool), solid.astype(bool)


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise ValueError("изделие не найдено: кадр пустой или фон не белый")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _dense_run(counts: np.ndarray, keep: float, max_trim: float) -> tuple[int, int]:
    """Границы плотной части профиля: срезает пологие края, но не больше `max_trim`."""
    n = len(counts)
    if n < 8:
        return 0, n
    window = max(3, n // 24 * 2 + 1)
    smooth = np.convolve(counts.astype(np.float32), np.ones(window, np.float32) / window,
                         mode="same")
    peak = float(smooth.max())
    if peak <= 0:
        return 0, n
    threshold = keep * peak
    lo, hi = 0, n
    while lo < n and smooth[lo] < threshold:
        lo += 1
    while hi > lo and smooth[hi - 1] < threshold:
        hi -= 1
    cap = int(n * max_trim)
    lo, hi = min(lo, cap), max(hi, n - cap)
    if hi - lo < n * 0.25:          # профиль не читается — рамку не трогаем
        return 0, n
    return lo, hi


@dataclass
class View:
    """Отдельный ракурс изделия в кадре (фас, профиль, вторая серьга ...)."""
    index: int
    bbox: tuple[int, int, int, int]
    area: int
    mask: np.ndarray

    @property
    def w(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def h(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def cx(self) -> float:
        return (self.bbox[0] + self.bbox[2]) / 2

    @property
    def cy(self) -> float:
        return (self.bbox[1] + self.bbox[3]) / 2


@dataclass
class Product:
    """Изделие, вписанное в итоговый холст, вместе с геометрией."""
    canvas: Image.Image
    thin: np.ndarray
    solid: np.ndarray
    bbox: tuple[int, int, int, int]
    views: list[View]
    source_size: tuple[int, int]
    scale: float

    @property
    def size(self) -> int:
        return self.canvas.width

    def keepout(self, pad: int = 0) -> np.ndarray:
        """Силуэт, расширенный на `pad` пикселей — зона, куда нельзя класть разметку."""
        if pad <= 0:
            return self.solid
        k = 2 * pad + 1
        return cv2.dilate(self.solid.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)

    def column_extent(self, x0: int, x1: int,
                      y0: int | None = None, y1: int | None = None) -> tuple[int, int] | None:
        """Вертикальный габарит изделия в полосе колонок [x0, x1).

        `y0`/`y1` ограничивают поиск строками: так габарит меряет именно вставку,
        а не весь металл, попавший в те же колонки.
        """
        top = max(0, y0) if y0 is not None else 0
        bottom = min(self.size, y1) if y1 is not None else self.size
        if bottom - top < 1:
            top, bottom = 0, self.size
        band = self.thin[top:bottom, max(0, x0):max(1, x1)]
        ys = np.where(band.any(axis=1))[0]
        return (int(ys.min()) + top, int(ys.max()) + top + 1) if len(ys) else None

    def ray_hit(self, start, end, overshoot: float = 0.0,
                within: tuple | None = None) -> tuple[float, float] | None:
        """Первая точка металла на луче `start` → `end`, с продлением на `overshoot`.

        Нужна, чтобы конец выноски садился ровно на силуэт: якорь, посчитанный по
        полосе строк, на наклонной кромке промахивается на её наклон, а марш вдоль
        самой линии попадает в контур при любом наклоне.

        `within` ограничивает область, где попадание засчитывается: для выноски к
        декоративному элементу это его рамка, иначе луч остановится на шинке,
        которая лежит между полкой и вставкой.
        """
        sx, sy = start
        ex, ey = end
        dx, dy = ex - sx, ey - sy
        span = math.hypot(dx, dy)
        if span < 1e-6:
            return None
        dx, dy = dx / span, dy / span
        for step in range(int(span + overshoot) + 1):
            x, y = sx + dx * step, sy + dy * step
            ix, iy = int(round(x)), int(round(y))
            if not (0 <= ix < self.size and 0 <= iy < self.size):
                continue
            if within and not (within[0] <= x <= within[2] and within[1] <= y <= within[3]):
                continue
            if self.thin[iy, ix]:
                return x, y
        return None

    def reach(self, start, direction, limit: float | None = None) -> float:
        """Насколько далеко от `start` вдоль `direction` тянется изделие.

        Нужна наклонным габаритам: размерную линию надо отодвинуть за силуэт,
        а по какой из осей он кончается — заранее не известно.
        """
        dx, dy = direction
        norm = math.hypot(dx, dy) or 1.0
        dx, dy = dx / norm, dy / norm
        limit = limit if limit is not None else self.size
        last = 0.0
        for step in range(int(limit) + 1):
            x, y = int(round(start[0] + dx * step)), int(round(start[1] + dy * step))
            if not (0 <= x < self.size and 0 <= y < self.size):
                break
            if self.thin[y, x]:
                last = float(step)
        return last

    def nearest_in_rect(self, target, center, axis, long: float, short: float,
                        pad: float = 0.06) -> tuple[float, float] | None:
        """Ближайший к `target` пиксель металла внутри повёрнутого прямоугольника.

        Так якорь выноски высоты гарантированно оказывается на самом декоративном
        элементе: не в пустоте, куда его заносило с угла описанной рамки, и не на
        шинке, которая подходит к декору вплотную.
        """
        ux, uy = axis
        nx, ny = -uy, ux
        half_long = long / 2 * (1 + pad)
        half_short = short / 2 * (1 + pad)
        reach = math.hypot(half_long, half_short)
        x0 = max(0, int(center[0] - reach))
        y0 = max(0, int(center[1] - reach))
        x1 = min(self.size, int(center[0] + reach) + 1)
        y1 = min(self.size, int(center[1] + reach) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None

        ys, xs = np.mgrid[y0:y1, x0:x1]
        dx = xs - center[0]
        dy = ys - center[1]
        along = dx * ux + dy * uy
        across = dx * nx + dy * ny
        inside = (np.abs(along) <= half_long) & (np.abs(across) <= half_short)
        metal = inside & self.thin[y0:y1, x0:x1]
        if not metal.any():
            return None
        distance = (xs - target[0]) ** 2 + (ys - target[1]) ** 2
        distance = np.where(metal, distance, np.inf)
        index = int(np.argmin(distance))
        return float(xs.flat[index]), float(ys.flat[index])

    def dense_box(self, box: tuple, keep: float = 0.40,
                  max_trim: float = 0.10) -> tuple[float, float, float, float]:
        """Поджимает рамку к плотной части изделия внутри неё.

        Модель регулярно прихватывает в рамку декоративного элемента плечо шинки:
        по краям рамки остаются колонки, где металла на просвет мало, а сама вставка
        даёт заметно более высокие столбцы. Поэтому края срезаются, пока профиль
        маски не поднимется до `keep` от своего максимума. Срез ограничен долей
        `max_trim` стороны — чтобы неверный профиль не съел рамку целиком.
        """
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(self.size, x1), min(self.size, y1)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return tuple(box)
        sub = self.thin[y0:y1, x0:x1]
        if not sub.any():
            return tuple(box)
        cx0, cx1 = _dense_run(sub.sum(axis=0), keep, max_trim)
        cy0, cy1 = _dense_run(sub.sum(axis=1), keep, max_trim)
        return (x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1)

    def row_extent(self, y0: int, y1: int,
                   x0: int | None = None, x1: int | None = None) -> tuple[int, int] | None:
        """Горизонтальный габарит изделия в полосе строк [y0, y1).

        `x0`/`x1` ограничивают поиск колонками: так якорь выноски садится на
        нужную деталь, а не на первый попавшийся металл в той же строке.
        """
        left = max(0, x0) if x0 is not None else 0
        right = min(self.size, x1) if x1 is not None else self.size
        if right - left < 1:
            left, right = 0, self.size
        band = self.thin[max(0, y0):max(1, y1), left:right]
        xs = np.where(band.any(axis=0))[0]
        return (int(xs.min()) + left, int(xs.max()) + left + 1) if len(xs) else None


def _split_views(solid: np.ndarray) -> list[View]:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(solid.astype(np.uint8), 8)
    views: list[View] = []
    floor = max(MIN_VIEW_FRAC * solid.size, 0.02 * max(
        (stats[i, cv2.CC_STAT_AREA] for i in range(1, count)), default=0))
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < floor:
            continue
        views.append(View(index=len(views), bbox=(int(x), int(y), int(x + w), int(y + h)),
                          area=int(area), mask=(labels == i)))
    views.sort(key=lambda v: v.bbox[0])
    for i, v in enumerate(views):
        v.index = i
    return views


def prepare(path: str | Path, canvas: int = CANVAS,
            max_frac: float = PRODUCT_MAX_FRAC) -> Product:
    """Обрезает поля, равномерно вписывает изделие в белый холст, считает маски.

    Изделие только переносится и масштабируется целиком — пропорции, цвет и
    геометрия не меняются.
    """
    src = Image.open(path).convert("RGB")
    rgb = np.asarray(src)
    _, solid = _masks(rgb)
    x0, y0, x1, y1 = _bbox(solid)

    crop = src.crop((x0, y0, x1, y1))
    scale = (max_frac * canvas) / max(crop.width, crop.height)
    target = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(target, Image.LANCZOS)

    sheet = Image.new("RGB", (canvas, canvas), BG)
    sheet.paste(crop, ((canvas - target[0]) // 2, (canvas - target[1]) // 2))

    thin, solid = _masks(np.asarray(sheet))
    return Product(
        canvas=sheet,
        thin=thin,
        solid=solid,
        bbox=_bbox(solid),
        views=_split_views(solid),
        source_size=src.size,
        scale=scale,
    )
