@echo off
set "DEEPSEEK_API_KEY=sk-f4a611dc64e94f6587ef1ccd169ee38a"

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
