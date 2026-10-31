"""Чтение задания: Рабочая_таблица.xlsx или Задачи/*.md."""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable, Iterator

from .config import STATUS_TODO

# Заголовки рабочей таблицы -> поля CatalogRow
XLSX_COLUMNS = {
    "ID": "id",
    "Номенклатура": "article",
    "Бренд": "brand",
    "Статус работы": "status",
    "Тип металла": "metal",
    "Высота декоративного элемента, мм": "height",
    "Длина декоративного элемента, мм": "length",
    "Ширина декоративного элемента, мм": "width",
    "Минимальная толщина шинки, мм": "band_t",
    "Минимальная ширина шинки, мм": "band_w",
    "Партия": "batch",
    "Файл результата": "out_name",
    "Комментарий": "comment",
    "Фото 1500": "url_1500",
    "Фото 1000": "url_1000",
}

NUMERIC = ("height", "length", "width", "band_t", "band_w")


@dataclass
class CatalogRow:
    id: str
    article: str = ""
    brand: str = ""
    metal: str = ""
    status: str = STATUS_TODO
    height: float = 0.0
    length: float = 0.0
    width: float = 0.0
    band_t: float = 0.0
    band_w: float = 0.0
    batch: str = ""
    out_name: str = ""
    comment: str | None = None
    url_1500: str | None = None
    url_1000: str | None = None
    row_index: int = 0          # 1-based номер строки в xlsx (для обратной записи)

    @property
    def urls(self) -> list[str]:
        return [u for u in (self.url_1500, self.url_1000) if u]

    @property
    def output_name(self) -> str:
        return self.out_name or f"{self.id}.png"

    def dims(self) -> dict[str, float]:
        return {k: getattr(self, k) for k in NUMERIC}

    def as_dict(self) -> dict:
        return asdict(self)


def _num(value) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).replace(",", ".").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group()) if match else 0.0


# ---------------------------------------------------------------- xlsx

def load_xlsx(path: str | Path) -> list[CatalogRow]:
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    rows = ws.iter_rows(values_only=True)
    header = [str(c).strip() if c is not None else "" for c in next(rows)]
    index = {XLSX_COLUMNS[h]: i for i, h in enumerate(header) if h in XLSX_COLUMNS}
    if "id" not in index:
        raise ValueError(f"{path}: не найден столбец ID (заголовки: {header})")

    out: list[CatalogRow] = []
    for n, raw in enumerate(rows, start=2):
        if not raw or raw[index["id"]] in (None, ""):
            continue
        kwargs = {}
        for field_name, col in index.items():
            value = raw[col] if col < len(raw) else None
            kwargs[field_name] = _num(value) if field_name in NUMERIC else (
                str(value).strip() if value is not None else None
            )
        kwargs["status"] = kwargs.get("status") or STATUS_TODO
        for key in ("article", "brand", "metal", "batch", "out_name"):
            kwargs[key] = kwargs.get(key) or ""
        out.append(CatalogRow(row_index=n, **kwargs))
    wb.close()
    return out


# ---------------------------------------------------------------- markdown

_MD_LINK = re.compile(r"\[(?P<label>1500|1000)\]\((?P<url>[^)]+)\)")


def load_markdown(path: str | Path) -> list[CatalogRow]:
    """Парсит таблицу «Фото и размеры» из Задачи/Партия_NN.md."""
    text = Path(path).read_text(encoding="utf-8")
    out: list[CatalogRow] = []
    for line in text.splitlines():
        if not line.startswith("S2-"):
            continue
        cells = [c.strip() for c in line.split("|")]
        if len(cells) < 9:
            continue
        links = dict(
            (m.group("label"), m.group("url")) for m in _MD_LINK.finditer(cells[7])
        )
        out.append(
            CatalogRow(
                id=cells[0],
                article=cells[1],
                height=_num(cells[2]),
                length=_num(cells[3]),
                width=_num(cells[4]),
                band_t=_num(cells[5]),
                band_w=_num(cells[6]),
                url_1500=links.get("1500"),
                url_1000=links.get("1000"),
                out_name=cells[8],
                batch=Path(path).stem,
            )
        )
    return out


def load(path: str | Path) -> list[CatalogRow]:
    path = Path(path)
    if path.suffix.lower() in (".xlsx", ".xlsm"):
        return load_xlsx(path)
    if path.suffix.lower() == ".md":
        return load_markdown(path)
    raise ValueError(f"Неизвестный формат задания: {path}")


# ---------------------------------------------------------------- фильтры

def select(
    rows: Iterable[CatalogRow],
    ids: Iterable[str] | None = None,
    batch: str | None = None,
    limit: int | None = None,
    redo: bool = False,
) -> list[CatalogRow]:
    """Отбирает строки под запуск: по ID, по партии, по количеству."""
    result: Iterator[CatalogRow] = iter(rows)
    if ids:
        wanted = {i.strip().upper() for i in ids}
        result = (r for r in result if r.id.upper() in wanted)
    if batch:
        needle = batch.strip().lower()
        result = (r for r in result if needle in (r.batch or "").lower())
    if not redo:
        result = (r for r in result if (r.status or STATUS_TODO) == STATUS_TODO)
    picked = list(result)
    return picked[:limit] if limit else picked
