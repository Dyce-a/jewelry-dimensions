@echo off
rem Двойной щелчок по этому файлу открывает редактор разметки.
rem Работает из любого места: %~dp0 — папка, где лежит сам файл.
chcp 65001 >nul
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Редактор.ps1" %*
if errorlevel 1 (
  echo.
  echo Редактор завершился с ошибкой. Окно останется открытым, чтобы её было видно.
  pause
)
