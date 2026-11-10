# PROSPERITY MARK-2 - Watchdog de Resurrección en PowerShell
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$Host.UI.RawUI.WindowTitle = "PROSPERITY MARK-2 Watchdog"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

$PythonExe = "python.exe"
if (Test-Path "$ScriptDir\..\.venv\Scripts\python.exe") {
    $PythonExe = (Resolve-Path "$ScriptDir\..\.venv\Scripts\python.exe").Path
} elseif (Test-Path "$ScriptDir\.venv\Scripts\python.exe") {
    $PythonExe = (Resolve-Path "$ScriptDir\.venv\Scripts\python.exe").Path
}

Write-Host "=====================================================================" -ForegroundColor Cyan
Write-Host " PROSPERITY MARK-2 - WATCHDOG ACTIVO (Recuperación Automática)" -ForegroundColor Green
Write-Host " Python: $PythonExe" -ForegroundColor Yellow
Write-Host " Directorio: $ScriptDir" -ForegroundColor Yellow
Write-Host "=====================================================================" -ForegroundColor Cyan

while ($true) {
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$Timestamp] Iniciando / Verificando PROSPERITY MARK-2..." -ForegroundColor Green
    
    try {
        & $PythonExe main_mark2.py
        $ExitCode = $LASTEXITCODE
        Write-Warning "[$Timestamp] MARK-2 se detuvo con código de salida: $ExitCode"
    } catch {
        Write-Error "[$Timestamp] Excepción en ejecución: $_"
    }
    
    Write-Host "Pausa de 5 segundos antes de reiniciar (Ctrl+C para salir)..." -ForegroundColor Gray
    Start-Sleep -Seconds 5
}
