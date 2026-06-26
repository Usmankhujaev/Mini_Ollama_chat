#!/usr/bin/env bash
# ============================================================================
#  Запуск ИИ-ассистента одной командой (macOS / Linux).
#
#      bash run.sh
#
#  Скрипт сам всё проверит и подскажет, чего не хватает:
#    1) есть ли Python 3
#    2) создаст виртуальное окружение и поставит зависимости
#    3) есть ли Ollama и запущен ли он (при необходимости запустит)
#    4) есть ли хотя бы одна модель (предложит скачать)
#    5) откроет приложение в браузере
#
#  Студенту НЕ нужно ничего настраивать вручную — просто запустите этот файл.
# ============================================================================
set -e
cd "$(dirname "$0")"

# Адрес Ollama. По умолчанию локальный; для удалённого сервера:
#   OLLAMA_URL="http://10.50.50.202:11434" bash run.sh
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

# Модели, которые ставим по умолчанию, если у студента ещё ничего нет.
CHAT_MODEL="llama3.2"            # текстовый чат
EMBED_MODEL="nomic-embed-text"  # семантический поиск по файлам (RAG)

say()  { printf "\n\033[1m%s\033[0m\n" "$1"; }   # жирный заголовок шага
ok()   { printf "  \033[32m[ OK ]\033[0m %s\n" "$1"; }
warn() { printf "  \033[33m[ !  ]\033[0m %s\n" "$1"; }
die()  { printf "\n\033[31m[ X  ] %s\033[0m\n" "$1"; exit 1; }

# ---------------------------------------------------------------- 1) Python
say "1/5 · Проверяю Python…"
if command -v python3 >/dev/null 2>&1; then
  PY=python3
elif command -v python >/dev/null 2>&1; then
  PY=python
else
  die "Python 3 не найден. Установите его с https://www.python.org/downloads/ и запустите снова."
fi
ok "$($PY --version)"

# ------------------------------------------------- 2) окружение и зависимости
say "2/5 · Готовлю окружение и зависимости…"
if [ ! -d .venv ]; then
  ok "создаю виртуальное окружение .venv"
  "$PY" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt
ok "зависимости установлены"

# ------------------------------------------------------------ 3) Ollama есть?
say "3/5 · Проверяю Ollama…"
if ! command -v ollama >/dev/null 2>&1; then
  warn "Ollama не установлен."
  echo "      Установите его (это бесплатно и локально):"
  echo "        macOS:  brew install ollama   — или скачайте с https://ollama.com/download"
  echo "        Linux:  curl -fsSL https://ollama.com/install.sh | sh"
  die "После установки Ollama запустите этот скрипт снова: bash run.sh"
fi
ok "Ollama установлен"

# --------------------------------------------------- 4) Ollama запущен? модели?
say "4/5 · Проверяю сервер Ollama и модели…"
# Поднят ли сервер (отвечает ли API). Если нет — пробуем запустить в фоне.
if ! curl -s -o /dev/null "$OLLAMA_URL/api/tags"; then
  warn "Сервер Ollama не отвечает — запускаю в фоне…"
  ollama serve >/tmp/ollama-run.log 2>&1 &
  for _ in $(seq 1 30); do
    sleep 1
    curl -s -o /dev/null "$OLLAMA_URL/api/tags" && break
  done
fi
if ! curl -s -o /dev/null "$OLLAMA_URL/api/tags"; then
  die "Не удалось подключиться к Ollama по $OLLAMA_URL. Запустите вручную: ollama serve"
fi
ok "сервер Ollama отвечает на $OLLAMA_URL"

# Есть ли хоть одна модель? Если нет — предлагаем скачать стандартный набор.
if ! ollama list 2>/dev/null | tail -n +2 | grep -q .; then
  warn "Модели ещё не скачаны."
  printf "      Скачать стандартный набор (%s + %s)? [Y/n] " "$CHAT_MODEL" "$EMBED_MODEL"
  read -r ans
  case "$ans" in
    [Nn]*) warn "Пропускаю. Скачайте модель вручную: ollama pull $CHAT_MODEL" ;;
    *)     ollama pull "$CHAT_MODEL"; ollama pull "$EMBED_MODEL" ;;
  esac
else
  ok "модели найдены"
fi

# ----------------------------------------------------------------- 5) запуск
say "5/5 · Запускаю приложение…"
echo "  Откроется в браузере: http://localhost:8501"
echo "  Остановить: Ctrl+C"
echo ""
streamlit run app.py
