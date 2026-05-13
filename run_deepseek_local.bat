@echo off
set "DEEPSEEK_API_KEY=sk-f83a1248e6fd4f5fa463fb891dbc2490"

start cmd /k "cd frontend && npm run dev"
timeout /t 5
python launcher.py --dev