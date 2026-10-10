# Lara, the AI receptionist. NOT YET TESTED: see "Docker" in the README for the known limits.
FROM python:3.12-slim

# uv gives fast, reproducible installs from uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
ENV UV_PYTHON_DOWNLOADS=never UV_COMPILE_BYTECODE=1

WORKDIR /app

# Dependencies first, so this layer is reused unless pyproject.toml or uv.lock change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .

EXPOSE 7860

# --host 0.0.0.0 is what makes the server reachable from outside the container.
# --no-sync stops uv trying to install anything at start-up.
CMD ["uv", "run", "--no-sync", "bot.py", "--host", "0.0.0.0"]