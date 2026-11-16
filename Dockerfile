FROM python:3.13.14-slim-trixie
COPY --from=ghcr.io/astral-sh/uv:0.10.12 /uv /uvx /usr/local/bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:${PATH}" \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    HF_HOME=/home/rag/.cache/huggingface

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev --no-editable \
    && groupadd --system rag \
    && useradd --system --gid rag --create-home --home-dir /home/rag rag \
    && mkdir -p /app/data/raw "$HF_HOME/hub" \
    && chown -R rag:rag /app/data /home/rag/.cache

USER rag
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD ["/app/.venv/bin/python", "-c", "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=4).close()"]
CMD ["rag", "serve", "--host", "0.0.0.0"]
