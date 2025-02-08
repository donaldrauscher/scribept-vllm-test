FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ADD . /app
WORKDIR /app

RUN uv sync --frozen

EXPOSE 8501

CMD ["uv", "run", "--", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]
