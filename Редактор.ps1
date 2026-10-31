<#
    Запуск локального редактора разметки.

        .\Редактор.ps1              — открыть на http://127.0.0.1:8765
        .\Редактор.ps1 -Port 9000   — другой порт
        .\Редактор.ps1 -NoOpen      — не открывать браузер
        .\Редактор.ps1 -Code нет    — запретить Опусу править код
        .\Редактор.ps1 -Code всё    — открыть ему весь проект, не только web/

    Скрипт сам находит Python, проверяет зависимости и поднимает сервер.
#>
[CmdletBinding()]
param(
    [int]$Port = 8765,
    [switch]$NoOpen,
    [ValidateSet("нет", "сайт", "всё")]
    [string]$Code = "сайт",
    [string]$Task = "данные\Рабочая_таблица.xlsx"
)

$ErrorActionPreference = "Stop"
# Консоль Windows по умолчанию в cp866: без этого кириллица в выводе рассыпается
$null = & chcp 65001
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
Set-Location -LiteralPath $PSScriptRoot

function Find-Python {
    foreach ($name in @("python", "py")) {
        $found = Get-Command $name -ErrorAction SilentlyContinue
        if ($null -ne $found) {
            if ($name -eq "py") { return @{ Exe = $found.Source; Args = @("-3") } }
            return @{ Exe = $found.Source; Args = @() }
        }
    }
    return $null
}

$python = Find-Python
if ($null -eq $python) {
    Write-Host "Python не найден. Поставьте его с python.org и откройте PowerShell заново." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "  РАЗМЕТКА ГАБАРИТОВ - редактор" -ForegroundColor DarkYellow
Write-Host "  ------------------------------" -ForegroundColor DarkGray

# Зависимости: ставим только те, которых не хватает
$probe = & $python.Exe @($python.Args + @("-c", "import PIL, cv2, numpy, requests, openpyxl")) 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Ставлю недостающие библиотеки..." -ForegroundColor DarkGray
    & $python.Exe @($python.Args + @("-m", "pip", "install", "-q", "-r", "requirements.txt"))
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  Не удалось поставить зависимости: $probe" -ForegroundColor Red
        exit 1
    }
}

if (-not (Test-Path -LiteralPath $Task)) {
    Write-Host "  Не нашёл рабочую таблицу: $Task" -ForegroundColor Red
    exit 1
}

$viewerArgs = @("viewer.py", "--port", $Port, "--task", $Task, "--code", $Code)
if ($NoOpen) { $viewerArgs += "--no-open" }

Write-Host "  Адрес: http://127.0.0.1:$Port/" -ForegroundColor Green
Write-Host "  Опусу разрешено править: $Code" -ForegroundColor DarkGray
Write-Host "  Остановить: Ctrl+C" -ForegroundColor DarkGray
Write-Host ""

& $python.Exe @($python.Args + $viewerArgs)
