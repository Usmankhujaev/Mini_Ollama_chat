#!/usr/bin/env bash
# Одна команда: создаёт окружение, ставит зависимости и запускает чатбот.
#   bash run.sh
set -e
cd "$(dirname "$0")"

# адрес вашего Ollama (локальный по умолчанию; для удалённого сервера поменяйте)
export OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

python3 -m venv .venv
source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo ""
echo "✅ Готово. Ollama: $OLLAMA_URL"
echo "🌐 Открываю http://localhost:8501 …"
echo ""
streamlit run app.py
