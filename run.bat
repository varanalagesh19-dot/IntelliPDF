@echo off
setlocal
cd /d "%~dp0"

set "VENV_DIR=.venv"

if not exist "%VENV_DIR%" (
    echo Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
)

if exist "%VENV_DIR%\Scripts\python.exe" (
    set "PY=%VENV_DIR%\Scripts\python.exe"
) else (
    set "PY=python"
)

if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env
        echo Created .env from .env.example. Please update your API keys.
    )
)

echo Starting IntelliPDF Backend (Port 8000)...
start "IntelliPDF Backend" "%PY%" -m uvicorn backend.main:app --reload --port 8000

echo Starting IntelliPDF Frontend (Port 8501)...
start "IntelliPDF Frontend" "%PY%" -m streamlit run frontend/app.py --server.port 8501

echo IntelliPDF is running:
echo   Frontend: http://localhost:8501
echo   Backend API: http://127.0.0.1:8000/docs
