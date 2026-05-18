@echo off
set "DEEPSEEK_API_KEY=sk-22497600591844cab54cef7ec091d48a"

start cmd /k "cd frontend && npm run dev"
timeout /t 5
python launcher.py --dev
set "APP_EXIT_CODE=%ERRORLEVEL%"
echo.
if not "%APP_EXIT_CODE%"=="0" (
    echo Backend exited with error code %APP_EXIT_CODE%.
) else (
    echo Backend exited.
)
echo Press any key to close this window.
pause >nul
