@echo off
chcp 65001 > nul
echo =====================================================================
echo  PROSPERITY MARK-2 - DESINSTALADOR DE AUTO-INICIO
echo =====================================================================
echo.
echo Eliminando la tarea programada "ProsperityMark2_Watchdog"...
schtasks /delete /tn "ProsperityMark2_Watchdog" /f
echo.
echo Listo.
pause
