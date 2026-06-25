@echo off
REM Одна команда для Windows: ставит зависимости и запускает чатбот.
REM   run.bat
cd /d "%~dp0"

if "%OLLAMA_URL%"=="" set OLLAMA_URL=http://localhost:11434

python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo.
echo Готово. Ollama: %OLLAMA_URL%
echo Открываю http://localhost:8501 ...
echo.
streamlit run app.py
