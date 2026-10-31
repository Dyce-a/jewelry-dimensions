#!/usr/bin/env python
"""Нанесение размерных габаритов на фото ювелирных изделий.

Примеры:
    python annotate.py --task "Рабочая_таблица.xlsx" --batch "Пилот"
    python annotate.py --task "Задачи/Партия_01.md" --limit 5 --debug
    python annotate.py --task "Рабочая_таблица.xlsx" --ids S2-001 S2-016 --overwrite
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from markup import catalog, report
from markup.config import API_KEY_ENV, CANVAS, DEFAULT_MODEL, STATUS_OK
from markup.pipeline import Options, run


def load_env(path: str | Path = ".env") -> None:
    """Подхватывает KEY=VALUE из .env рядом со скриптом, не перетирая окружение."""
    path = Path(path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Размерные габариты на фото ювелирных изделий",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--task", required=True,
                   help="Рабочая_таблица.xlsx или Задачи/Партия_NN.md")
    p.add_argument("--out", default="out", help="каталог для готовых PNG")
    p.add_argument("--cache", default="cache", help="каталог кэша исходных фото")
    p.add_argument("--ids", nargs="*", help="обработать только эти ID (S2-001 ...)")
    p.add_argument("--batch", help="обработать только эту партию (например «Пилот»)")
    p.add_argument("--limit", type=int, help="ограничить число позиций")
    p.add_argument("--redo", action="store_true",
                   help="брать строки в любом статусе, а не только «Не начато»")
    p.add_argument("--overwrite", action="store_true",
                   help="перерисовывать, даже если файл результата уже существует")
    p.add_argument("--model", default=DEFAULT_MODEL, help="модель зрения в ProxyAPI")
    p.add_argument("--no-vision", action="store_true",
                   help="без обращения к модели, только геометрическая раскладка")
    p.add_argument("--replan", action="store_true",
                   help="заново спросить модель, игнорируя кэш планов в cache/plans")
    p.add_argument("--no-refine", action="store_true",
                   help="без второго прохода, уточняющего рамку декоративного элемента")
    p.add_argument("--canvas", type=int, default=CANVAS, help="сторона итогового PNG, px")
    p.add_argument("--workers", type=int, default=4, help="параллельных потоков")
    p.add_argument("--write-xlsx", metavar="PATH",
                   help="проставить статусы в этой рабочей таблице")
    p.add_argument("--csv", metavar="PATH", default="out/реестр.csv",
                   help="реестр прогона для приложения к отчёту")
    p.add_argument("--debug", action="store_true",
                   help="сохранять план разметки в out/_debug/*.json")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    load_env(Path(__file__).with_name(".env"))

    rows = catalog.load(args.task)
    picked = catalog.select(rows, ids=args.ids, batch=args.batch,
                            limit=args.limit, redo=args.redo)
    if not picked:
        print("Под фильтр не попала ни одна строка.", file=sys.stderr)
        return 1

    api_key = os.environ.get(API_KEY_ENV)
    use_vision = not args.no_vision
    if use_vision and not api_key:
        print(f"Ключ {API_KEY_ENV} не задан — работаю без модели, по геометрии.",
              file=sys.stderr)
        use_vision = False

    opts = Options(
        out_dir=Path(args.out), cache_dir=Path(args.cache), model=args.model,
        api_key=api_key, use_vision=use_vision, refine=not args.no_refine,
        replan=args.replan,
        canvas=args.canvas,
        workers=max(1, args.workers), overwrite=args.overwrite, debug=args.debug,
    )

    done = 0
    total = len(picked)

    def progress(result):
        nonlocal done
        done += 1
        mark = "ok " if result.status == STATUS_OK else "!! "
        tail = f" — {result.comment}" if result.comment else ""
        print(f"[{done}/{total}] {mark}{result.row.id} {result.status}{tail}", flush=True)

    results = run(picked, opts, on_done=progress)

    if args.csv:
        report.write_csv(args.csv, results)
    if args.write_xlsx:
        changed = report.write_xlsx(args.write_xlsx, results)
        print(f"В рабочую таблицу записано строк: {changed}")

    print()
    print(report.summary(results))
    return 0 if all(r.status == STATUS_OK for r in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
