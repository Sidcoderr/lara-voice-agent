# Lara, the AI receptionist.
# Tested on Docker Desktop for Windows (WSL 2). See "Docker" in the README for what has and has not been tried.
# The release name is pinned (bookworm) so the system package names below stay valid.
FROM python:3.12-slim-bookworm

# Libraries some compiled Python packages expect (OpenCV and the ONNX models Pipecat uses).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# uv gives fast, reproducible installs from uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependencies first, so this layer is reused unless pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

EXPOSE 7860

# Docker reports "healthy" once the server answers.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/staff', timeout=4)" || exit 1

# --host 0.0.0.0 is what makes the server reachable from outside the container.
# --no-sync stops uv trying to install anything at start-up.
CMD ["uv", "run", "--no-sync", "bot.py", "--host", "0.0.0.0"]