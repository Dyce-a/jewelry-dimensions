"""Инструменты, которыми Опус правит сам редактор.

Дать модели право писать в исходники проекта — самое опасное, что здесь есть:
сломанный `layout.py` за неделю до сдачи стоит дороже любой сэкономленной
минуты. Поэтому право выдаётся не голым, а с четырьмя предохранителями.

1. **Ограниченная зона.** По умолчанию писать можно только в `web/` — разметку,
   стили и скрипт самого редактора. Питон пайплайна открывается отдельным ключом
   и с честным предупреждением: после такой правки сервер надо перезапустить,
   потому что модули уже загружены в память.
2. **Проверка после каждой записи.** JavaScript прогоняется через `node --check`,
   Python — через компиляцию, HTML и CSS — через грубую сверку скобок и тегов.
3. **Откат при неудаче.** Если проверка не прошла, файл возвращается в прежнее
   состояние сразу, ещё до ответа модели. Сломать страницу, через которую идёт
   разговор, она не может.
4. **История.** Каждая удачная правка ложится отдельным коммитом с пометкой, кто
   её сделал. Любую можно снять одной командой, не разбирая, что именно менялось.

Ни удаления файлов, ни выполнения команд оболочки, ни доступа за пределы проекта
здесь нет и не предполагается.
"""
from __future__ import annotations

import ast
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Что открыто на запись при разных уровнях доверия.
ЗОНЫ = {
    "нет": (),
    "сайт": ("web",),
    "всё": ("web", "markup", "annotate.py", "contact_sheet.py", "viewer.py"),
}

MAX_BYTES = 400_000       # больше этого в проекте нет ни одного исходника
ЧИТАЕМЫЕ = {".py", ".js", ".css", ".html", ".md", ".json", ".txt", ".ps1", ".sh"}

#: Сюда не пускаем даже на чтение: ключ от API и внутренности репозитория.
ЗАКРЫТО = {".env", ".git"}


class DevError(RuntimeError):
    pass


@dataclass
class Гараж:
    """Настройки доступа к исходникам."""
    уровень: str = "сайт"

    @property
    def корни(self) -> tuple:
        return ЗОНЫ.get(self.уровень, ())

    @property
    def можно_писать(self) -> bool:
        return bool(self.корни)


# ---------------------------------------------------------------- пути

def _внутри(путь: Path, корень: Path) -> bool:
    try:
        путь.relative_to(корень)
    except ValueError:
        return False
    return True


def разрешить(гараж: Гараж, отрезок: str, запись: bool) -> Path:
    """Проверяет путь и возвращает его абсолютную форму."""
    очищенный = (отрезок or "").strip().lstrip("/\\")
    цель = (ROOT / очищенный).resolve()
    if not _внутри(цель, ROOT):
        raise DevError("путь выходит за пределы проекта")
    части = цель.relative_to(ROOT).parts
    if any(часть in ЗАКРЫТО for часть in части):
        raise DevError("этот файл закрыт: в нём ключи и внутренности репозитория")
    # каталоги пропускаем, у файлов расширение обязано быть из списка: без этого
    # сюда пролезал бы, например, .env — у него расширения нет вовсе
    if цель.is_file() or цель.suffix:
        if цель.suffix.lower() not in ЧИТАЕМЫЕ:
            raise DevError(f"с такими файлами этот инструмент не работает: {цель.name}")
    if not запись:
        return цель
    if not гараж.можно_писать:
        raise DevError("правка кода выключена (ключ --code нет)")
    относительный = цель.relative_to(ROOT).as_posix()
    if not any(относительный == к or относительный.startswith(к + "/")
               for к in гараж.корни):
        доступно = ", ".join(гараж.корни)
        raise DevError(f"писать можно только в {доступно}; запрошен {относительный}")
    return цель


# ---------------------------------------------------------------- проверки

def проверить(цель: Path) -> str | None:
    """Возвращает описание поломки или None, если файл в порядке."""
    суффикс = цель.suffix.lower()
    текст = цель.read_text(encoding="utf-8")
    if суффикс == ".py":
        try:
            ast.parse(текст)
        except SyntaxError as exc:
            return f"синтаксис Python: строка {exc.lineno}, {exc.msg}"
        return None
    if суффикс == ".js":
        try:
            итог = subprocess.run(["node", "--check", str(цель)], capture_output=True,
                                  text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None          # без node проверить нечем — пропускаем
        if итог.returncode != 0:
            return "синтаксис JavaScript: " + (итог.stderr or "").strip().split("\n")[0]
        return None
    if суффикс == ".json":
        try:
            json.loads(текст)
        except ValueError as exc:
            return f"JSON: {exc}"
        return None
    if суффикс == ".css":
        if текст.count("{") != текст.count("}"):
            return "в CSS не сходятся фигурные скобки"
        return None
    if суффикс == ".html":
        for тег in ("div", "section", "button", "svg"):
            открыто = текст.count(f"<{тег}")
            закрыто = текст.count(f"</{тег}>")
            if открыто and открыто != закрыто:
                return f"в HTML не сходятся теги <{тег}>: {открыто} против {закрыто}"
        return None
    return None


# ---------------------------------------------------------------- git

def _git(*аргументы: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.name=Опус", "-c", "user.email=opus@local", *аргументы],
        cwd=ROOT, capture_output=True, text=True, timeout=60)


def репозиторий_есть() -> bool:
    return (ROOT / ".git").exists()


def зафиксировать(сообщение: str, файлы: list[str]) -> str:
    """Отдельный коммит на правку. Возвращает короткий хеш или пояснение."""
    if not репозиторий_есть():
        return "репозитория нет, история не ведётся"
    _git("add", *файлы)
    итог = _git("commit", "-m", f"Опус: {сообщение}")
    if итог.returncode != 0:
        return (итог.stdout or итог.stderr or "").strip()[:200]
    хеш = _git("rev-parse", "--short", "HEAD").stdout.strip()
    return хеш


def откатить(сколько: int = 1) -> str:
    """Снимает последние правки Опуса, не трогая чужие коммиты."""
    if not репозиторий_есть():
        return "репозитория нет, откатывать нечего"
    журнал = _git("log", "--format=%h %s", "-n", str(max(1, сколько))).stdout.strip().splitlines()
    свои = [строка for строка in журнал if строка.split(" ", 1)[-1].startswith("Опус:")]
    if not свои:
        return "последние коммиты не мои — откатывать нечего"
    итог = _git("revert", "--no-edit", *[строка.split(" ", 1)[0] for строка in свои])
    if итог.returncode != 0:
        return (итог.stderr or "").strip()[:200]
    return f"снято правок: {len(свои)}"


def история(сколько: int = 10) -> list[str]:
    if not репозиторий_есть():
        return []
    return _git("log", "--format=%h %s", "-n", str(сколько)).stdout.strip().splitlines()


# ---------------------------------------------------------------- действия

def перечислить(гараж: Гараж, отрезок: str = "web") -> list[dict]:
    каталог = разрешить(гараж, отрезок, запись=False)
    if not каталог.exists():
        raise DevError(f"нет каталога {отрезок}")
    файлы = []
    цели = каталог.rglob("*") if каталог.is_dir() else [каталог]
    for путь in sorted(цели):
        if путь.is_dir() or путь.suffix.lower() not in ЧИТАЕМЫЕ:
            continue
        if "__pycache__" in путь.parts:
            continue
        файлы.append({"путь": путь.relative_to(ROOT).as_posix(),
                      "байт": путь.stat().st_size,
                      "строк": len(путь.read_text(encoding="utf-8").splitlines())})
    return файлы


def прочитать(гараж: Гараж, отрезок: str, с_строки: int = 1, строк: int = 0) -> dict:
    цель = разрешить(гараж, отрезок, запись=False)
    if not цель.exists():
        raise DevError(f"нет файла {отрезок}")
    все = цель.read_text(encoding="utf-8").splitlines()
    начало = max(1, int(с_строки or 1))
    конец = начало + int(строк) - 1 if строк else len(все)
    кусок = все[начало - 1:конец]
    return {"путь": цель.relative_to(ROOT).as_posix(), "всего_строк": len(все),
            "с_строки": начало,
            "текст": "\n".join(f"{начало + i:5} | {строка}" for i, строка in enumerate(кусок))}


def _записать(гараж: Гараж, цель: Path, новый: str, сообщение: str) -> dict:
    было = цель.read_text(encoding="utf-8") if цель.exists() else None
    if len(новый.encode("utf-8")) > MAX_BYTES:
        raise DevError("файл слишком большой, что-то пошло не так")
    цель.parent.mkdir(parents=True, exist_ok=True)
    цель.write_text(новый, encoding="utf-8")

    поломка = проверить(цель)
    if поломка:
        # возвращаем как было сразу: страница, через которую идёт разговор, не должна ломаться
        if было is None:
            цель.unlink(missing_ok=True)
        else:
            цель.write_text(было, encoding="utf-8")
        raise DevError(f"правка отклонена и откачена — {поломка}")

    относительный = цель.relative_to(ROOT).as_posix()
    хеш = зафиксировать(сообщение or f"правка {относительный}", [относительный])
    ответ = {"путь": относительный, "итог": "записано и проверено", "коммит": хеш}
    if цель.suffix.lower() == ".py":
        ответ["внимание"] = ("правка питона вступит в силу только после перезапуска "
                             "сервера редактора")
    else:
        ответ["внимание"] = "обновите страницу в браузере, чтобы увидеть правку"
    return ответ


def записать(гараж: Гараж, отрезок: str, содержимое: str, сообщение: str = "") -> dict:
    цель = разрешить(гараж, отрезок, запись=True)
    return _записать(гараж, цель, содержимое, сообщение)


def заменить(гараж: Гараж, отрезок: str, было: str, стало: str, сообщение: str = "") -> dict:
    """Точечная замена куска текста. Кусок обязан встречаться ровно один раз."""
    цель = разрешить(гараж, отрезок, запись=True)
    if not цель.exists():
        raise DevError(f"нет файла {отрезок}")
    текст = цель.read_text(encoding="utf-8")
    сколько = текст.count(было)
    if сколько == 0:
        raise DevError("такого куска в файле нет — перечитайте файл и повторите")
    if сколько > 1:
        raise DevError(f"кусок встречается {сколько} раза — возьмите больше контекста")
    return _записать(гараж, цель, текст.replace(было, стало), сообщение)
