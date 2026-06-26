@echo off
REM ==========================================================================
REM  Запуск ИИ-ассистента одной командой (Windows).
REM
REM      Дважды кликните по run.bat  —  или в консоли:  run.bat
REM
REM  Скрипт сам проверит Python, поставит зависимости, проверит Ollama
REM  и модели, затем откроет приложение в браузере.
REM  Студенту НЕ нужно ничего настраивать вручную.
REM ==========================================================================
setlocal
cd /d "%~dp0"
chcp 65001 >nul

REM Адрес Ollama (локальный по умолчанию; для удалённого сервера задайте свой).
if "%OLLAMA_URL%"=="" set OLLAMA_URL=http://localhost:11434

set CHAT_MODEL=llama3.2
set EMBED_MODEL=nomic-embed-text

REM ---------------------------------------------------------------- 1) Python
echo.
echo 1/5 - Проверяю Python...
where python >nul 2>&1
if errorlevel 1 (
  echo.
  echo [X] Python не найден. Установите его с https://www.python.org/downloads/
  echo     ВАЖНО: при установке поставьте галочку "Add Python to PATH".
  echo     Затем запустите run.bat снова.
  pause
  exit /b 1
)
python --version

REM ------------------------------------------------- 2) окружение и зависимости
echo.
echo 2/5 - Готовлю окружение и зависимости...
if not exist .venv (
  echo   создаю виртуальное окружение .venv
  python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install -q --upgrade pip
pip install -q -r requirements.txt
echo   зависимости установлены

REM ------------------------------------------------------------ 3) Ollama есть?
echo.
echo 3/5 - Проверяю Ollama...
where ollama >nul 2>&1
if errorlevel 1 (
  echo.
  echo [!] Ollama не установлен.
  echo     Скачайте и установите его с https://ollama.com/download
  echo     После установки запустите run.bat снова.
  pause
  exit /b 1
)
echo   Ollama установлен

REM --------------------------------------------------- 4) Ollama запущен? модели?
echo.
echo 4/5 - Проверяю сервер Ollama и модели...
curl -s -o nul "%OLLAMA_URL%/api/tags"
if errorlevel 1 (
  echo   сервер не отвечает - запускаю в фоне...
  start "" /b ollama serve
  REM ждём, пока поднимется API
  for /l %%i in (1,1,30) do (
    timeout /t 1 /nobreak >nul
    curl -s -o nul "%OLLAMA_URL%/api/tags" && goto :server_up
  )
)
:server_up
curl -s -o nul "%OLLAMA_URL%/api/tags"
if errorlevel 1 (
  echo.
  echo [X] Не удалось подключиться к Ollama по %OLLAMA_URL%.
  echo     Откройте отдельное окно и выполните:  ollama serve
  pause
  exit /b 1
)
echo   сервер Ollama отвечает на %OLLAMA_URL%

REM Есть ли хоть одна модель?
for /f "skip=1 delims=" %%m in ('ollama list 2^>nul') do goto :have_models
echo   [!] Модели ещё не скачаны.
set /p ans="      Скачать стандартный набор (%CHAT_MODEL% + %EMBED_MODEL%)? [Y/n] "
if /i "%ans%"=="n" (
  echo   Пропускаю. Скачайте модель вручную:  ollama pull %CHAT_MODEL%
) else (
  ollama pull %CHAT_MODEL%
  ollama pull %EMBED_MODEL%
)
:have_models

REM ----------------------------------------------------------------- 5) запуск
echo.
echo 5/5 - Запускаю приложение...
echo   Откроется в браузере: http://localhost:8501
echo   Остановить: Ctrl+C
echo.
streamlit run app.py
