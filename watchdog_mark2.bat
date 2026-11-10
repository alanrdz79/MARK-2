@echo off
chcp 65001 > nul
title PROSPERITY MARK-2 - Supervisor Watchdog Resiliente

echo =====================================================================
echo  PROSPERITY MARK-2 - WATCHDOG DE SUPERVISION Y AUTO-RECUPERACION
echo =====================================================================
echo Directorio: %~dp0
cd /d "%~dp0"

:: Detectar ejecutable de Python preferido
set PYTHON_BIN=
if exist "..\.venv\Scripts\python.exe" (
    set PYTHON_BIN=..\.venv\Scripts\python.exe
) else if exist ".venv\Scripts\python.exe" (
    set PYTHON_BIN=.venv\Scripts\python.exe
) else (
    set PYTHON_BIN=python.exe
)

echo Usando entorno Python: %PYTHON_BIN%
echo.

:WATCH_LOOP
echo ---------------------------------------------------------------------
echo [%DATE% %TIME%] Iniciando / Reanudando PROSPERITY MARK-2...
echo ---------------------------------------------------------------------

"%PYTHON_BIN%" main_mark2.py

set EXIT_CODE=%ERRORLEVEL%
echo.
echo [%DATE% %TIME%] AVISO: El proceso finalizo con codigo de salida: %EXIT_CODE%

:: Pausa de reanudación
echo [%DATE% %TIME%] Reanudando en 5 segundos (Presione Ctrl+C para cancelar)...
timeout /t 5 /nobreak > nul
goto WATCH_LOOP
