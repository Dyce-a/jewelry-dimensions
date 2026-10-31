"""Сквозная обработка одной позиции и партии."""
from __future__ import annotations

import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import CatalogRow
from .config import CANVAS, DEFAULT_MODEL, STATUS_ANGLE, STATUS_ERROR, STATUS_OK
from .imaging import FetchError, fetch, prepare
from .layout import solve
from .planner import Plan, build
from .render import Renderer


def _plan_cache_path(row: CatalogRow, opts: "Options") -> Path:
    return opts.cache_dir / "plans" / f"{row.id}.json"


def _load_plan(row: CatalogRow, opts: "Options", canvas: int) -> Plan | None:
    """Достаёт сохранённый план обращений к модели.

    План — самая дорогая часть прогона (3–4 запроса к модели), но от способа
    отрисовки он не зависит. Поэтому правки раскладки и линий перепроверяются
    на кэше бесплатно, а модель дёргается только для новых позиций.
    """
    path = _plan_cache_path(row, opts)
    if not path.exists():
        return None
    try:
        plan = Plan.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError, KeyError, TypeError):
        return None
    # правку человека не перебивает даже --replan: она дороже любого ответа модели
    if opts.replan and not (plan.manual or plan.proposed):
        return None
    return plan.rescale(canvas)


def _save_plan(row: CatalogRow, opts: "Options", plan: Plan) -> None:
    path = _plan_cache_path(row, opts)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2),
                        encoding="utf-8")
    except OSError:
        pass


@dataclass
class Options:
    out_dir: Path = Path("out")
    cache_dir: Path = Path("cache")
    model: str = DEFAULT_MODEL
    api_key: str | None = None
    use_vision: bool = True
    canvas: int = CANVAS
    workers: int = 4
    overwrite: bool = False
    refine: bool = True
    replan: bool = False        # заново спросить модель, игнорируя кэш планов
    debug: bool = False


@dataclass
class Result:
    row: CatalogRow
    status: str = STATUS_ERROR
    comment: str = ""
    out_path: Path | None = None
    plan: Plan | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


def process(row: CatalogRow, opts: Options) -> Result:
    result = Result(row=row)
    out_path = opts.out_dir / row.output_name
    if out_path.exists() and not opts.overwrite:
        result.status = STATUS_OK
        result.out_path = out_path
        return result

    try:
        photo = fetch(row.urls, opts.cache_dir / "photos", row.id)
    except FetchError as exc:
        result.comment = f"фото недоступно: {exc}"
        return result

    try:
        product = prepare(photo, canvas=opts.canvas)
    except Exception as exc:                          # noqa: BLE001
        result.comment = f"не удалось отделить изделие от фона: {exc}"
        if opts.debug:
            result.comment += "\n" + traceback.format_exc()
        return result

    plan = _load_plan(row, opts, product.size)
    warning = None
    if plan is None:
        plan, warning = build(product, row, model=opts.model, api_key=opts.api_key,
                              use_vision=opts.use_vision, refine=opts.refine,
                              cache=opts.cache_dir)
        if plan.source == "vision":
            _save_plan(row, opts, plan)
    result.plan = plan

    if plan.problem:
        result.status = STATUS_ANGLE
        result.comment = plan.problem
        return result

    try:
        image, _boxes, issues = solve(product, plan, row, Renderer(size=opts.canvas),
                                      cache=opts.cache_dir)
    except Exception as exc:                          # noqa: BLE001
        result.comment = f"ошибка раскладки: {exc}"
        if opts.debug:
            result.comment += "\n" + traceback.format_exc()
        return result

    result.issues = list(plan.notes) + issues
    opts.out_dir.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG", optimize=True)
    result.out_path = out_path

    blocking = [i for i in issues if "накрывает изделие" in i]
    if blocking:
        result.status = STATUS_ANGLE
        result.comment = "; ".join(dict.fromkeys(blocking))
    else:
        result.status = STATUS_OK
        notes = [n for n in ([warning] if warning else []) + result.issues if n]
        result.comment = "; ".join(dict.fromkeys(notes))

    if opts.debug:
        debug_dir = opts.out_dir / "_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        (debug_dir / f"{row.id}.json").write_text(
            json.dumps({"plan": plan.raw or {"source": plan.source},
                        "decor_box": plan.decor_box, "decor_source": plan.decor_source,
                        "axes": plan.axes, "leader_dim": plan.leader_dim,
                        "notes": plan.notes, "issues": issues},
                       ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def run(rows: list[CatalogRow], opts: Options, on_done=None) -> list[Result]:
    """Обрабатывает список позиций; сеть и модель распараллелены по потокам."""
    results: list[Result] = []
    if opts.workers <= 1:
        for row in rows:
            result = process(row, opts)
            results.append(result)
            if on_done:
                on_done(result)
        return results

    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        for result in pool.map(lambda r: process(r, opts), rows):
            results.append(result)
            if on_done:
                on_done(result)
    return results
