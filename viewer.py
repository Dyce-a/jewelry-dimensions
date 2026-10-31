#!/usr/bin/env python
"""Локальный редактор разметки: галерея готовых кадров и правка геометрии руками.

Запуск:
    python viewer.py                      # http://127.0.0.1:8765
    python viewer.py --port 9000 --no-open

Сервер намеренно на стандартной библиотеке: новых зависимостей у проекта не
появляется, а нагрузка тут — одна вкладка браузера.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import threading
import traceback
import webbrowser
from collections import OrderedDict
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from PIL import Image

from markup import catalog
from markup.config import (API_KEY_ENV, CANVAS, DEFAULT_MODEL, STATUS_ANGLE,
                           STATUS_ERROR, STATUS_OK, STATUS_TODO)
from markup.inspect import canvas_preview, describe, load
from markup.layout import find_decor, solve
from markup.planner import Plan, build, clean_tweak
from markup.render import Renderer

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
THUMB = 360


# ---------------------------------------------------------------- рабочая область

class Workspace:
    """Каталог, кэш подготовленных изделий и запись правок на диск."""

    def __init__(self, task: Path, out: Path, cache: Path, canvas: int,
                 код: str = "сайт") -> None:
        from markup.devtools import Гараж

        self.task, self.out, self.cache, self.canvas = task, out, cache, canvas
        self.гараж = Гараж(код)
        self.lock = threading.Lock()
        self.products: OrderedDict[str, object] = OrderedDict()
        #: эшелоны, на которых остановилась последняя отрисовка кадра. Схема в
        #: редакторе рисуется по ним же — иначе человек тянул бы мышью линию,
        #: стоящую в PNG совсем на другой высоте
        self.echelons: dict[str, dict] = {}
        self.reload()

    # -- каталог ------------------------------------------------------

    def reload(self) -> None:
        from markup import uploads

        rows = catalog.load(self.task)
        # свои загрузки идут первыми: человек только что их принёс и смотреть
        # будет на них, а не листать девятьсот кадров поставщика
        rows = uploads.rows(self.cache) + rows
        self.rows = {row.id: row for row in rows}
        self.order = [row.id for row in rows]

    def plan_path(self, item_id: str) -> Path:
        return self.cache / "plans" / f"{item_id}.json"

    def out_path(self, item_id: str) -> Path | None:
        row = self.rows.get(item_id)
        if not row:
            return None
        path = self.out / row.output_name
        return path if path.exists() else None

    # -- тяжёлые объекты ----------------------------------------------

    def product(self, item_id: str):
        """Подготовленное изделие; последние восемь держим в памяти."""
        with self.lock:
            if item_id in self.products:
                self.products.move_to_end(item_id)
                return self.products[item_id]
        product, _ = load(self.rows[item_id], self.cache,
                          self.plan_path(item_id), self.canvas)
        with self.lock:
            self.products[item_id] = product
            while len(self.products) > 8:
                self.products.popitem(last=False)
        return product

    def plan(self, item_id: str) -> Plan:
        row = self.rows[item_id]
        path = self.plan_path(item_id)
        if path.exists():
            try:
                return Plan.from_dict(json.loads(path.read_text(encoding="utf-8"))).rescale(
                    self.canvas)
            except (OSError, ValueError, KeyError, TypeError):
                pass
        product = self.product(item_id)
        plan, _warning = build(product, row, model=DEFAULT_MODEL,
                               api_key=os.environ.get(API_KEY_ENV),
                               use_vision=bool(os.environ.get(API_KEY_ENV)),
                               cache=self.cache)
        self.save_plan(item_id, plan)
        return plan

    def save_plan(self, item_id: str, plan: Plan) -> None:
        path = self.plan_path(item_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(plan.to_dict(), ensure_ascii=False, indent=2,
                                   default=float), encoding="utf-8")

    # -- отрисовка -----------------------------------------------------

    def render(self, item_id: str) -> tuple[Path, list[str]]:
        row = self.rows[item_id]
        product = self.product(item_id)
        plan = self.plan(item_id)
        эшелоны: dict = {}
        image, _boxes, issues = solve(product, plan, row, Renderer(size=self.canvas),
                                      cache=self.cache, levels_out=эшелоны)
        self.echelons[item_id] = эшелоны
        self.out.mkdir(parents=True, exist_ok=True)
        path = self.out / row.output_name
        image.save(path, format="PNG", optimize=True)
        thumb = self.cache / "thumbs" / f"{item_id}.jpg"
        thumb.unlink(missing_ok=True)
        return path, list(issues)

    # -- статусы -------------------------------------------------------

    def marks_path(self) -> Path:
        return self.out / "_контроль" / "правки.json"

    def marks(self) -> dict:
        path = self.marks_path()
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return {}
        return {}

    def set_mark(self, item_id: str, status: str, comment: str) -> None:
        marks = self.marks()
        marks[item_id] = {"status": status, "comment": comment,
                          "изменено": datetime.now().isoformat(timespec="seconds")}
        path = self.marks_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(marks, ensure_ascii=False, indent=2), encoding="utf-8")

    def drop_mark(self, item_id: str) -> None:
        """Забывает статус позиции: своя загрузка убрана — статусу не за чем стоять."""
        marks = self.marks()
        if marks.pop(item_id, None) is None:
            return
        self.marks_path().write_text(json.dumps(marks, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    def flush_xlsx(self) -> int:
        """Переносит накопленные правки статусов в рабочую таблицу."""
        import openpyxl

        marks = self.marks()
        if not marks:
            return 0
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        shutil.copy2(self.task, self.task.with_name(
            f"{self.task.stem}.backup-{stamp}{self.task.suffix}"))
        book = openpyxl.load_workbook(self.task)
        sheet = book.worksheets[0]
        header = [str(c.value).strip() if c.value is not None else "" for c in sheet[1]]
        status_col = header.index("Статус работы") + 1
        comment_col = header.index("Комментарий") + 1
        changed = 0
        for item_id, mark in marks.items():
            row = self.rows.get(item_id)
            if not row or not row.row_index:
                continue
            sheet.cell(row=row.row_index, column=status_col).value = mark["status"]
            sheet.cell(row=row.row_index, column=comment_col).value = mark.get("comment") or None
            changed += 1
        book.save(self.task)
        return changed


# ---------------------------------------------------------------- данные для страниц

def item_summary(space: Workspace, item_id: str, marks: dict) -> dict:
    from markup import uploads

    row = space.rows[item_id]
    out = space.out_path(item_id)
    mark = marks.get(item_id)
    plan_path = space.plan_path(item_id)
    manual = proposed = False
    tweaks: list = []
    if plan_path.exists():
        try:
            данные = json.loads(plan_path.read_text(encoding="utf-8"))
            manual = bool(данные.get("manual"))
            proposed = bool(данные.get("proposed"))
            tweaks = list((данные.get("tweaks") or {}).keys())
        except (OSError, ValueError):
            pass
    return {
        "id": item_id,
        "article": row.article,
        "batch": row.batch,
        "metal": row.metal,
        "status": (mark or {}).get("status") or row.status or STATUS_TODO,
        "comment": (mark or {}).get("comment") or row.comment or "",
        "dims": row.dims(),
        "ready": bool(out),
        "manual": manual,
        "proposed": proposed,
        "planned": plan_path.exists(),
        "upload": uploads.is_upload(item_id),
        "tweaked": sorted(tweaks),
    }


def item_detail(space: Workspace, item_id: str) -> dict:
    row = space.rows[item_id]
    product = space.product(item_id)
    plan = space.plan(item_id)
    canvas_preview(product, space.cache / "холст" / f"{item_id}.jpg")
    data = describe(row, plan, product, space.cache, space.echelons.get(item_id))
    if plan.manual and plan.decor_lock:
        # что дал бы автомат без правки — нужно, чтобы разбор сравнивал
        # рамку человека именно с машинной, а не саму с собой
        свободный = Plan.from_dict(plan.to_dict())
        свободный.decor_lock, свободный.decor_rect, свободный.manual = "", None, False
        авто, _ = find_decor(product, свободный, row)
        data["auto_decor"] = {
            "center": [round(v, 1) for v in авто.center],
            "long": round(авто.long, 1), "short": round(авто.short, 1),
            "angle": round(авто.angle, 1), "source": авто.source,
        }
    marks = space.marks()
    data["item"] = item_summary(space, item_id, marks)
    data["plan"] = {
        "item_type": plan.item_type, "view": plan.view, "views_count": plan.views_count,
        "axes": plan.axes, "leader_dim": plan.leader_dim,
        "band_present": plan.band_present, "band_side": plan.band_side,
        "band_width_visible": plan.band_width_visible,
        "band_cuts": dict(plan.band_cuts),
        "problem": plan.problem, "source": plan.source,
        "decor_source": plan.decor_source, "manual": plan.manual,
        "decor_lock": plan.decor_lock,
    }
    return data


# ---------------------------------------------------------------- правки размеров

def apply_tweaks(space: Workspace, item_id: str, plan: Plan, raw: dict) -> set:
    """Кладёт правки размеров в план и возвращает, какие размеры тронуты.

    Присланный набор считается полным: размер, которого в нём нет, вернулся под
    автомат. Так «снять правку» не требует отдельного маршрута — достаточно
    убрать её на кадре.

    Якорь выноски высоты идёт особым путём — в свой журнал (`cache/якоря.json`):
    по нему автомат учится ставить стрелку туда же на похожих ракурсах, и терять
    это ради единообразия не стоит.
    """
    from markup import anchors, habits
    from markup.layout import find_decor

    очищенные = {}
    for ключ, поправка in (raw or {}).items():
        if not isinstance(поправка, dict):
            continue
        поля = clean_tweak(поправка, space.canvas)
        if поля:
            очищенные[ключ] = поля

    правка = очищенные.get(plan.leader_dim) or {}
    if "anchor" in правка:
        точка = [float(правка["anchor"][0]), float(правка["anchor"][1])]
        plan.leader_anchor = точка
        plan.leader_lock = True
        правка.pop("anchor")
        if not правка:
            очищенные.pop(plan.leader_dim, None)
        try:
            элемент, _ = find_decor(space.product(item_id), plan, space.rows[item_id])
            anchors.remember(space.cache, item_id, plan.item_type, plan.view,
                             точка, элемент)
        except Exception:                       # noqa: BLE001 — журнал необязателен
            pass

    снятые = set(plan.tweaks or {}) - set(очищенные)
    if снятые:
        habits.forget(space.cache, item_id, снятые)
    plan.tweaks = очищенные
    return set(очищенные)


def remember_habits(space: Workspace, item_id: str, plan: Plan, ключи) -> None:
    """Записывает привычку по тем размерам, которые человек только что подвинул."""
    from markup import habits
    from markup.layout import compose, find_decor

    try:
        row = space.rows[item_id]
        product = space.product(item_id)
        элемент, _ = find_decor(product, plan, row)
        лист = compose(product, plan, row, space.echelons.get(item_id, {}),
                       элемент, space.cache)
        habits.remember(space.cache, item_id, plan, product, лист, set(ключи))
    except Exception:                           # noqa: BLE001 — журнал необязателен
        pass


# ---------------------------------------------------------------- своё фото

def принять_фото(space: Workspace, body: dict) -> dict:
    """Загруженный снимок: позиция каталога, план, разметка — за один заход.

    Числа берутся из формы и дальше не обсуждаются, ровно как из рабочей таблицы:
    модель зрения и здесь работает измерителем, а не источником размеров.
    """
    from markup import uploads

    try:
        row = uploads.add(space.cache, body.get("файл") or "",
                          body.get("размеры") or {}, body.get("имя") or "")
    except uploads.UploadError as exc:
        return {"ошибка": str(exc)}

    space.reload()
    try:
        _path, issues = space.render(row.id)
    except Exception as exc:                    # noqa: BLE001 — причину показываем человеку
        return {"ошибка": f"{exc}", "id": row.id}
    # статус ставится сразу: кадр уже в галерее, и цикл на этом замкнулся
    space.set_mark(row.id, STATUS_ANGLE if issues else STATUS_OK,
                   "; ".join(issues)[:400])
    return {"готово": True, "id": row.id, "замечания": issues,
            "разбор": item_detail(space, row.id)}


# ---------------------------------------------------------------- HTTP

class Handler(BaseHTTPRequestHandler):
    server_version = "MarkupViewer"
    space: Workspace

    def log_message(self, fmt, *args):        # тише в консоли
        pass

    def thumb(self, item_id: str) -> Path:
        return make_thumb(self.space, item_id)

    # -- ответы -------------------------------------------------------

    def _send(self, code: int, body: bytes, kind: str, cache: bool = False) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, payload, code: int = 200) -> None:
        self._send(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _file(self, path: Path, kind: str, cache: bool = False) -> None:
        if not path.exists():
            self._json({"ошибка": f"нет файла {path.name}"}, 404)
            return
        self._send(200, path.read_bytes(), kind, cache)

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    # -- маршруты -----------------------------------------------------

    def do_GET(self) -> None:
        try:
            self.route_get(urlparse(self.path))
        except Exception as exc:                        # noqa: BLE001
            self._json({"ошибка": str(exc), "след": traceback.format_exc()}, 500)

    def do_POST(self) -> None:
        try:
            self.route_post(urlparse(self.path))
        except Exception as exc:                        # noqa: BLE001
            self._json({"ошибка": str(exc), "след": traceback.format_exc()}, 500)

    def route_get(self, url) -> None:
        path = unquote(url.path)
        space = self.space

        if path == "/":
            self._file(WEB / "index.html", "text/html; charset=utf-8")
            return
        if path.startswith("/static/fonts/"):
            # шрифт берём из assets: тот же Montserrat, что и на самих выкладках
            font = (ROOT / "assets" / "fonts" / Path(path).name).resolve()
            if font.parent == (ROOT / "assets" / "fonts").resolve():
                self._file(font, "font/ttf", cache=True)
            else:
                self._json({"ошибка": "нельзя"}, 403)
            return
        if path.startswith("/static/"):
            name = path[len("/static/"):]
            kinds = {".css": "text/css; charset=utf-8",
                     ".js": "text/javascript; charset=utf-8",
                     ".ttf": "font/ttf", ".svg": "image/svg+xml"}
            target = (WEB / name).resolve()
            if WEB.resolve() not in target.parents:
                self._json({"ошибка": "нельзя"}, 403)
                return
            # код и стили не кэшируем: иначе после правки браузер показывает старую версию
            self._file(target, kinds.get(target.suffix, "application/octet-stream"))
            return

        if path == "/api/setup":
            from markup import devtools
            self._json({"правка_кода": space.гараж.уровень,
                        "зона": list(space.гараж.корни),
                        "репозиторий": devtools.репозиторий_есть(),
                        "усилия": ["low", "medium", "high", "xhigh", "max"]})
            return

        if path == "/api/learning":
            from markup import advisor, anchors, habits
            self._json({"якоря": anchors.summary(space.cache),
                        "привычки": habits.summary(space.cache),
                        "правила": advisor.load_rules(space.cache)})
            return

        if path == "/api/flush":
            self._json({"записано": space.flush_xlsx()})
            return

        if path == "/api/items":
            query = parse_qs(url.query)
            marks = space.marks()
            items = [item_summary(space, i, marks) for i in space.order]
            batch = (query.get("batch") or [""])[0]
            status = (query.get("status") or [""])[0]
            text = (query.get("q") or [""])[0].strip().lower()
            only = (query.get("only") or [""])[0]
            if batch:
                items = [i for i in items if i["batch"] == batch]
            if status:
                items = [i for i in items if i["status"] == status]
            if only == "ready":
                items = [i for i in items if i["ready"]]
            elif only == "manual":
                items = [i for i in items if i["manual"]]
            elif only == "proposed":
                items = [i for i in items if i["proposed"]]
            elif only == "upload":
                items = [i for i in items if i["upload"]]
            if text:
                items = [i for i in items
                         if text in i["id"].lower() or text in (i["article"] or "").lower()]
            self._json({
                "всего": len(space.order),
                "показано": len(items),
                "партии": sorted({space.rows[i].batch for i in space.order if space.rows[i].batch}),
                "статусы": [STATUS_OK, STATUS_ANGLE, STATUS_ERROR, STATUS_TODO],
                "позиции": items,
            })
            return

        match = re.fullmatch(r"/api/item/([\w\-]+)", path)
        if match:
            self._json(item_detail(space, match.group(1)))
            return

        match = re.fullmatch(r"/media/(out|thumb|canvas)/([\w\-]+)", path)
        if match:
            kind, item_id = match.groups()
            if kind == "out":
                target = space.out_path(item_id)
                if not target:
                    self._json({"ошибка": "кадр ещё не размечен"}, 404)
                    return
                self._file(target, "image/png")
            elif kind == "thumb":
                self._file(self.thumb(item_id), "image/jpeg", cache=True)
            else:
                product = space.product(item_id)
                self._file(canvas_preview(product, space.cache / "холст" / f"{item_id}.jpg"),
                           "image/jpeg", cache=True)
            return

        self._json({"ошибка": "нет такого адреса", "путь": path}, 404)

    def route_post(self, url) -> None:
        path = unquote(url.path)
        space = self.space

        if path == "/api/chat":
            from markup.assistant import chat
            тело = self._body()
            self._json(chat(space, тело.get("история") or [], тело.get("id"),
                            тело.get("усилие") or "medium"))
            return

        if path == "/api/spread/apply":
            from markup import spread
            body = self._body()
            сделано = spread.propose(space, list(body.get("ids") or []),
                                     dict(body.get("поправка") or {}),
                                     str(body.get("образец") or ""))
            self._json({"перерисовано": сделано})
            return

        if path == "/api/spread/decide":
            body = self._body()
            принять = bool(body.get("accept"))
            изменено = 0
            for item_id in body.get("ids") or []:
                plan = space.plan(item_id)
                if not plan.proposed:
                    continue
                if принять:
                    plan.proposed, plan.manual = "", True
                else:
                    plan.proposed = ""
                    plan.decor_lock, plan.decor_rect, plan.manual = "", None, False
                space.save_plan(item_id, plan)
                if not принять:
                    space.render(item_id)
                изменено += 1
            self._json({"обработано": изменено})
            return

        if path == "/api/upload":
            self._json(принять_фото(space, self._body()))
            return

        match = re.fullmatch(r"/api/item/([\w\-]+)/(save|reset|render|status|analyse|spread|remove)", path)
        if not match:
            self._json({"ошибка": "нет такого адреса", "путь": path}, 404)
            return
        item_id, action = match.groups()
        body = self._body()

        if action == "save":
            plan = space.plan(item_id)
            rect = body.get("rect")
            if rect:
                plan.decor_rect = {"center": [float(rect["cx"]), float(rect["cy"])],
                                   "long": float(rect["long"]), "short": float(rect["short"]),
                                   "angle": float(rect["angle"])}
                plan.decor_lock = "oriented"
                half = max(plan.decor_rect["long"], plan.decor_rect["short"]) / 2
                plan.decor_box = (plan.decor_rect["center"][0] - half,
                                  plan.decor_rect["center"][1] - half,
                                  plan.decor_rect["center"][0] + half,
                                  plan.decor_rect["center"][1] + half)
            if body.get("leader_anchor"):
                точка = [float(body["leader_anchor"][0]), float(body["leader_anchor"][1])]
                plan.leader_anchor = точка
                plan.leader_lock = True
                # отметка человека идёт в журнал: по нескольким таким отметкам
                # автомат учится ставить якорь туда же на похожих ракурсах
                try:
                    from markup import anchors
                    from markup.layout import find_decor
                    элемент, _ = find_decor(space.product(item_id), plan, space.rows[item_id])
                    anchors.remember(space.cache, item_id, plan.item_type, plan.view,
                                     точка, элемент)
                except Exception:                       # noqa: BLE001 — журнал необязателен
                    pass
            if body.get("band_point"):
                plan.band_point = (float(body["band_point"][0]), float(body["band_point"][1]))
            if "band_width_visible" in body:
                plan.band_width_visible = bool(body["band_width_visible"])
            if body.get("band_cuts"):
                plan.band_cuts = {**plan.band_cuts,
                                  **{k: ("h" if v == "h" else "v")
                                     for k, v in body["band_cuts"].items()
                                     if k in ("band_w", "band_t")}}
            if body.get("axes"):
                plan.axes = dict(body["axes"])
            if "tweaks" in body:
                тронуты = apply_tweaks(space, item_id, plan, body["tweaks"])
            else:
                тронуты = set()
            plan.manual = True
            space.save_plan(item_id, plan)
            _path, issues = space.render(item_id)
            if тронуты:
                remember_habits(space, item_id, plan, тронуты)
            self._json({"готово": True, "замечания": issues,
                        "разбор": item_detail(space, item_id)})
            return

        if action == "reset":
            from markup import habits

            plan = space.plan(item_id)
            plan.manual = False
            plan.decor_lock = ""
            plan.decor_rect = None
            plan.leader_lock = False
            plan.leader_anchor = None
            plan.tweaks = {}
            # снятая правка больше ничего не подтверждает: держать её в журнале
            # привычек значит учить автомат на том, от чего человек отказался
            habits.forget(space.cache, item_id)
            space.save_plan(item_id, plan)
            _path, issues = space.render(item_id)
            self._json({"готово": True, "замечания": issues,
                        "разбор": item_detail(space, item_id)})
            return

        if action == "render":
            _path, issues = space.render(item_id)
            self._json({"готово": True, "замечания": issues,
                        "разбор": item_detail(space, item_id)})
            return

        if action == "status":
            space.set_mark(item_id, body.get("status") or STATUS_OK,
                           body.get("comment") or "")
            self._json({"готово": True})
            return

        if action == "analyse":
            from markup.advisor import explain
            self._json(explain(space, item_id, body))
            return

        if action == "remove":
            from markup import habits, uploads

            убрано = uploads.remove(space.cache, item_id, space.out)
            habits.forget(space.cache, item_id)
            space.drop_mark(item_id)
            space.echelons.pop(item_id, None)
            space.reload()
            self._json({"готово": убрано})
            return

        if action == "spread":
            from markup import spread
            итог = spread.run(space, item_id, int(body.get("limit") or spread.DEFAULT_LIMIT),
                              verify_with_model=body.get("verify", True))
            for ключ in ("кандидаты", "подходят", "не_подходят"):
                if итог.get(ключ):
                    итог[ключ] = [{"id": i, "article": space.rows[i].article}
                                  for i in итог[ключ] if i in space.rows]
            self._json(итог)
            return


def make_thumb(space: Workspace, item_id: str) -> Path:
    path = space.cache / "thumbs" / f"{item_id}.jpg"
    if path.exists():
        return path
    source = space.out_path(item_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not source:
        Image.new("RGB", (THUMB, THUMB), (245, 244, 239)).save(path, quality=80)
        return path
    image = Image.open(source).convert("RGB")
    image.thumbnail((THUMB, THUMB), Image.LANCZOS)
    image.save(path, format="JPEG", quality=82)
    return path



# ---------------------------------------------------------------- запуск

def load_env(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Локальный редактор разметки")
    parser.add_argument("--task", default="данные/Рабочая_таблица.xlsx")
    parser.add_argument("--out", default="out")
    parser.add_argument("--cache", default="cache")
    parser.add_argument("--canvas", type=int, default=CANVAS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true", help="не открывать браузер")
    parser.add_argument("--code", default="сайт", choices=["нет", "сайт", "всё"],
                        help="что Опусу разрешено править: ничего, только web/ или весь проект")
    args = parser.parse_args(argv)

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    load_env(ROOT / ".env")

    Handler.space = Workspace(Path(args.task), Path(args.out), Path(args.cache),
                              args.canvas, args.code)
    address = f"http://{args.host}:{args.port}/"
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Редактор разметки: {address}")
    зона = ", ".join(Handler.space.гараж.корни) or "ничего"
    print(f"Позиций в таблице: {len(Handler.space.order)}. "
          f"Опусу разрешено править: {зона}. Остановить — Ctrl+C.")
    if not args.no_open:
        threading.Timer(0.7, lambda: webbrowser.open(address)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
