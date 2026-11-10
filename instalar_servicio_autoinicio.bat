@echo off
chcp 65001 > nul
echo =====================================================================
echo  PROSPERITY MARK-2 - INSTALADOR DE AUTO-INICIO EN WINDOWS
echo =====================================================================
echo.
echo Este script creara una tarea programada en Windows para que MARK-2
echo arranque automaticamente al encender o reiniciar la computadora.
echo.

set SCRIPT_PATH=%~dp0watchdog_mark2.bat

echo Registrando tarea programada "ProsperityMark2_Watchdog"...
schtasks /create /tn "ProsperityMark2_Watchdog" /tr "\"%SCRIPT_PATH%\"" /sc onstart /ru SYSTEM /f /rl HIGHEST

if %ERRORLEVEL% EQU 0 (
    echo.
    echo =====================================================================
    echo  EXITO: Tarea programada creada correctamente.
    echo  MARK-2 se ejecutara automaticamente cada vez que arranque Windows.
    echo =====================================================================
) else (
    echo.
    echo AVISO: Si fallo por permisos de administrador, se intentara como usuario actual...
    schtasks /create /tn "ProsperityMark2_Watchdog" /tr "\"%SCRIPT_PATH%\"" /sc onlogon /f
)
pause
