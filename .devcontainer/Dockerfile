FROM mcr.microsoft.com/vscode/devcontainers/python:3.11
COPY --from=ghcr.io/astral-sh/uv:0.6.2 /uv /uvx /bin/
ARG USERNAME=vscode

RUN apt-get update && \
  rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

COPY . /app

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv && uv sync

ENV VENV_DIR=.venv
RUN chown -R $USERNAME $VENV_DIR && chmod -R 755 $VENV_DIR

ENV IBIS_PROJECT=.
RUN chown -R $USERNAME $IBIS_PROJECT

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"

SHELL ["/bin/bash", "-c", "source .venv/bin/activate"]

ENTRYPOINT []

USER $USERNAME