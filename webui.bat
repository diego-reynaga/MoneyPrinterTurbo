@echo off
setlocal
set "CURRENT_DIR=%CD%"
echo ***** Directorio actual: %CURRENT_DIR% *****
set "PYTHONPATH=%CURRENT_DIR%"

rem set HF_ENDPOINT=https://hf-mirror.com

if not defined MPT_WEBUI_HOST set "MPT_WEBUI_HOST=127.0.0.1"
if not defined MPT_WEBUI_PORT set "MPT_WEBUI_PORT=8501"

set "STREAMLIT_CMD="
if exist "%CURRENT_DIR%\.venv\Scripts\python.exe" (
    set "STREAMLIT_CMD="%CURRENT_DIR%\.venv\Scripts\python.exe" -m streamlit"
) else if exist "%CURRENT_DIR%\lib\python\python.exe" (
    set "STREAMLIT_CMD="%CURRENT_DIR%\lib\python\python.exe" -m streamlit"
) else (
    where uv >nul 2>nul
    if not errorlevel 1 set "STREAMLIT_CMD=uv run streamlit"
)

if not defined STREAMLIT_CMD (
    where streamlit >nul 2>nul
    if not errorlevel 1 (
        echo ***** Aviso: se usa el streamlit del PATH. Si fallan las dependencias, ejecuta primero 'uv sync --frozen'. *****
        set "STREAMLIT_CMD=streamlit"
    )
)

if not defined STREAMLIT_CMD (
    echo ***** No se encontro el Python del proyecto, ni uv, ni streamlit. Instala primero las dependencias. *****
    pause
    exit /b 1
)

set "SELECTED_WEBUI_PORT="
for /f %%P in ('powershell -NoProfile -ExecutionPolicy Bypass -Command "$hostAddress=$null; foreach ($address in [Net.Dns]::GetHostAddresses($env:MPT_WEBUI_HOST)) { if ($address.AddressFamily -eq [Net.Sockets.AddressFamily]::InterNetwork) { $hostAddress=$address; break } }; if ($null -eq $hostAddress) { exit 1 }; $preferred=[int]$env:MPT_WEBUI_PORT; $candidates=New-Object System.Collections.Generic.List[int]; $candidates.Add($preferred); foreach ($candidate in 8502..8599) { if ($candidate -ne $preferred) { $candidates.Add($candidate) } }; foreach ($port in $candidates) { $socket=[Net.Sockets.Socket]::new([Net.Sockets.AddressFamily]::InterNetwork,[Net.Sockets.SocketType]::Stream,[Net.Sockets.ProtocolType]::Tcp); try { $socket.Bind([Net.IPEndPoint]::new($hostAddress,$port)); $socket.Close(); Write-Output $port; exit 0 } catch { try { $socket.Close() } catch {} } }; exit 1"') do set "SELECTED_WEBUI_PORT=%%P"

if not defined SELECTED_WEBUI_PORT (
    echo ***** No hay ningun puerto libre para la WebUI entre 8501 y 8599 en %MPT_WEBUI_HOST%. *****
    echo ***** Si Windows indica WinError 10013, revisa los puertos reservados: netsh interface ipv4 show excludedportrange protocol=tcp *****
    pause
    exit /b 1
)

if not "%SELECTED_WEBUI_PORT%"=="%MPT_WEBUI_PORT%" (
    echo ***** El puerto %MPT_WEBUI_PORT% no esta disponible; se usa %SELECTED_WEBUI_PORT% en su lugar. *****
)
set "MPT_WEBUI_PORT=%SELECTED_WEBUI_PORT%"

echo ***** Direccion de la WebUI: http://%MPT_WEBUI_HOST%:%MPT_WEBUI_PORT% *****
%STREAMLIT_CMD% run .\webui\Main.py --server.address=%MPT_WEBUI_HOST% --server.port=%MPT_WEBUI_PORT% --browser.serverAddress=%MPT_WEBUI_HOST% --browser.gatherUsageStats=False --client.toolbarMode=minimal --logger.hideWelcomeMessage=True --server.showEmailPrompt=False --server.enableCORS=True
