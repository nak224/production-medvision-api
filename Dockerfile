FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /usr/local/bin/uv
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    MEDVISION_CHECKPOINT=/models/model.pt \
    MPLCONFIGDIR=/tmp/matplotlib
COPY --chown=10001:10001 pyproject.toml uv.lock README.md ./
COPY --chown=10001:10001 src ./src
COPY --chown=10001:10001 api ./api
RUN --mount=type=secret,id=ca_bundle \
    if [ -f /run/secrets/ca_bundle ]; then export SSL_CERT_FILE=/run/secrets/ca_bundle; fi; \
    uv sync --frozen --no-dev --no-editable --no-cache && \
    useradd --create-home --uid 10001 app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
