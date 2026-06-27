# Мини-чатбот на Streamlit + Ollama
FROM python:3.12-slim

# Streamlit и сеть
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    # из контейнера Ollama хоста доступен по host.docker.internal
    OLLAMA_URL=http://host.docker.internal:11434

WORKDIR /app

# зависимости ставим отдельным слоем — кэшируются, пока requirements.txt не меняется
COPY requirements.txt .
RUN pip install -r requirements.txt

# код приложения
COPY app.py .

# каталог для сохранённых сессий (можно примонтировать томом)
RUN mkdir -p /app/.sessions
VOLUME ["/app/.sessions"]

EXPOSE 8501

# простой healthcheck встроенного эндпоинта Streamlit
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').status==200 else 1)"

CMD ["streamlit", "run", "app.py"]
