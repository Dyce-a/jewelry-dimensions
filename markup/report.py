"""Запись статусов обратно в Рабочая_таблица.xlsx и сводка по прогону."""
from __future__ import annotations

import csv
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from .catalog import XLSX_COLUMNS
from .config import STATUS_OK
from .pipeline import Result

STATUS_FIELD = "Статус работы"
COMMENT_FIELD = "Комментарий"


def write_xlsx(path: str | Path, results: list[Result], backup: bool = True) -> int:
    """Проставляет статус и комментарий в строках рабочей таблицы. Возвращает число правок."""
    import openpyxl

    path = Path(path)
    by_row = {r.row.row_index: r for r in results if r.row.row_index}
    if not by_row:
        return 0
    if backup:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(path, path.with_name(f"{path.stem}.backup-{stamp}{path.suffix}"))

    wb = openpyxl.load_workbook(path)
    ws = wb.worksheets[0]
    header = [str(c.value).strip() if c.value is not None else "" for c in ws[1]]
    try:
        status_col = header.index(STATUS_FIELD) + 1
        comment_col = header.index(COMMENT_FIELD) + 1
    except ValueError as exc:
        raise ValueError(f"{path}: не найдены столбцы «{STATUS_FIELD}»/«{COMMENT_FIELD}»") from exc

    changed = 0
    for row_index, result in by_row.items():
        ws.cell(row=row_index, column=status_col).value = result.status
        ws.cell(row=row_index, column=comment_col).value = result.comment or None
        changed += 1
    wb.save(path)
    return changed


def write_csv(path: str | Path, results: list[Result]) -> None:
    """Реестр прогона: по строке на позицию — для приложения к отчёту."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["ID", "Номенклатура", "Партия", "Статус", "Файл",
                         "Тип изделия", "Ракурс", "Источник плана", "Замечания"])
        for r in results:
            plan = r.plan
            writer.writerow([
                r.row.id, r.row.article, r.row.batch, r.status,
                r.out_path.name if r.out_path else "",
                plan.item_type if plan else "",
                plan.view if plan else "",
                plan.source if plan else "",
                "; ".join(r.issues),
            ])


def summary(results: list[Result]) -> str:
    counts = Counter(r.status for r in results)
    total = len(results)
    accepted = counts.get(STATUS_OK, 0)
    share = f"{accepted / total * 100:.1f}%" if total else "—"
    lines = [f"Обработано: {total}", f"Принято: {accepted} ({share})"]
    for status, count in counts.most_common():
        if status != STATUS_OK:
            lines.append(f"{status}: {count}")
    defects = Counter(i for r in results for i in r.issues)
    if defects:
        lines.append("Замечания:")
        lines += [f"  {count} x {text}" for text, count in defects.most_common(8)]
    return "\n".join(lines)
