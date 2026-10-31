"""Свои фотографии: приём снимка от человека и позиция каталога под него.

Рабочая таблица описывает девятьсот кадров поставщика, но редактору нужен и
обратный ход: человек кладёт собственное фото кольца, вводит те же пять чисел — и
получает размеченный кадр. Пайплайн для этого менять не нужно, потому что он
работает не с таблицей, а со строкой каталога: id, пять размеров и ссылка на
снимок. Модуль и делает такую строку — только ссылкой служит путь к файлу на
диске, а не адрес в облаке поставщика (`imaging.fetch` понимает и его).

Числа по-прежнему берутся из формы, а не из кадра: модель зрения остаётся
измерителем и в этом пути тоже. Загруженное фото не переписывается — оно только
приводится к JPEG, как и всё, что приходит от поставщика.

Список загрузок лежит в `cache/загрузки/каталог.json` и подмешивается к строкам
рабочей таблицы при каждом чтении каталога: перезапуск редактора их не теряет.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import re
from datetime import datetime
from pathlib import Path

from PIL import Image

from .catalog import CatalogRow, NUMERIC
from .config import STATUS_TODO

BATCH = "Мои загрузки"
PREFIX = "Загрузка"
MAX_BYTES = 30 * 1024 * 1024      # больше тридцати мегабайт студийное фото не весит
SIDE = 2000                       # до какой стороны ужимаем снимок при приёме
FILE = "каталог.json"


class UploadError(ValueError):
    pass


def store(cache: Path) -> Path:
    return Path(cache) / "загрузки"


def _catalog_path(cache: Path) -> Path:
    return store(cache) / FILE


def _read(cache: Path) -> list[dict]:
    path = _catalog_path(cache)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _write(cache: Path, записи: list[dict]) -> None:
    path = _catalog_path(cache)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(записи, ensure_ascii=False, indent=2), encoding="utf-8")


def rows(cache: Path) -> list[CatalogRow]:
    """Загруженные позиции как обычные строки каталога."""
    готовые = []
    for запись in _read(cache):
        поля = {ключ: запись.get(ключ) for ключ in
                ("id", "article", "brand", "metal", "batch", "out_name", "comment",
                 "url_1500", "url_1000")}
        поля = {ключ: значение for ключ, значение in поля.items() if значение is not None}
        поля.update({ключ: float(запись.get(ключ) or 0.0) for ключ in NUMERIC})
        готовые.append(CatalogRow(status=запись.get("status") or STATUS_TODO, **поля))
    return готовые


def _next_id(записи: list[dict]) -> str:
    номера = [int(m.group(1)) for m in
              (re.fullmatch(rf"{PREFIX}-(\d+)", з.get("id", "")) for з in записи) if m]
    return f"{PREFIX}-{max(номера, default=0) + 1:03d}"


def _decode(данные: str) -> bytes:
    """Достаёт байты из data-URL, каким его отдаёт браузер."""
    if not данные:
        raise UploadError("фото не приложено")
    хвост = данные.split("base64,", 1)[-1]
    try:
        сырое = base64.b64decode(хвост, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise UploadError(f"фото не разбирается: {exc}") from exc
    if len(сырое) > MAX_BYTES:
        raise UploadError("файл больше 30 МБ — ужмите его перед загрузкой")
    if len(сырое) < 1024:
        raise UploadError("файл слишком маленький — это не фотография")
    return сырое


def _save_photo(path: Path, сырое: bytes) -> None:
    """Кладёт снимок на диск в JPEG. Изделие при этом не перерисовывается.

    Ужимается только сторона: холст пайплайна всё равно 2000 px, а телефонный
    снимок на 6000 px тратит память впустую. Пропорции остаются, кроп не
    делается — «изделие не искажается» относится и к своим фотографиям.
    """
    try:
        снимок = Image.open(io.BytesIO(сырое))
        снимок.load()
    except Exception as exc:                 # noqa: BLE001 — формат приносит человек
        raise UploadError(f"это не изображение ({exc})") from exc
    if снимок.mode != "RGB":
        снимок = снимок.convert("RGB")
    if max(снимок.size) > SIDE:
        снимок.thumbnail((SIDE, SIDE), Image.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    снимок.save(path, format="JPEG", quality=94)


def add(cache: Path, файл: str, размеры: dict, article: str = "") -> CatalogRow:
    """Принимает фото и пять чисел, возвращает готовую строку каталога."""
    значения = {}
    for ключ in NUMERIC:
        try:
            значения[ключ] = float(str(размеры.get(ключ, 0)).replace(",", "."))
        except (TypeError, ValueError):
            raise UploadError(f"размер «{ключ}» — не число") from None
        if значения[ключ] < 0:
            raise UploadError(f"размер «{ключ}» отрицательный")
    if значения["length"] <= 0 or значения["width"] <= 0:
        raise UploadError("длина и ширина декоративного элемента обязательны: "
                          "без них нечего показывать габаритами")

    сырое = _decode(файл)
    записи = _read(cache)
    item_id = _next_id(записи)
    фото = store(cache) / f"{item_id}.jpg"
    _save_photo(фото, сырое)

    запись = {
        "id": item_id,
        "article": (article or "").strip()[:120] or "своё фото",
        "batch": BATCH,
        "status": STATUS_TODO,
        "out_name": f"{item_id}.png",
        "url_1500": str(фото.resolve()),
        "когда": datetime.now().isoformat(timespec="seconds"),
        **значения,
    }
    записи.append(запись)
    _write(cache, записи)
    return rows(cache)[-1]


def is_upload(item_id: str) -> bool:
    return bool(re.fullmatch(rf"{PREFIX}-\d+", item_id or ""))


def remove(cache: Path, item_id: str, out: Path) -> bool:
    """Убирает загрузку целиком: снимок, план, готовый кадр, миниатюру.

    Трогаются только файлы самой загрузки. Фотографии поставщика и рабочая
    таблица тут ни при чём, и удалить их этим путём нельзя.
    """
    if not is_upload(item_id):
        raise UploadError("удалять можно только свои загрузки")
    записи = _read(cache)
    остаток = [з for з in записи if з.get("id") != item_id]
    if len(остаток) == len(записи):
        return False
    _write(cache, остаток)
    cache = Path(cache)
    for путь in (store(cache) / f"{item_id}.jpg",
                 cache / "plans" / f"{item_id}.json",
                 cache / "thumbs" / f"{item_id}.jpg",
                 cache / "холст" / f"{item_id}.jpg",
                 Path(out) / f"{item_id}.png"):
        путь.unlink(missing_ok=True)
    return True
