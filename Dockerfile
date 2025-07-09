# Stage 1: Build package with Poetry. This needs both main and dev dependencies so,
# while we can build the project from this, we don't want to include all the
# dependencies from this stage.
FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_CACHE_DIR="/opt/poetry-cache"
ENV POETRY_VENV_IN_PROJECT=1
ENV POETRY_NO_INTERACTION=1
ENV POETRY_VERSION=1.8.3

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./
# Don't create venv since we're using the container as the environment
RUN poetry config virtualenvs.create false
RUN poetry install --only=main,dev --no-root

COPY src/pm25ml/ ./pm25ml/
COPY README.md ./
RUN poetry build



# Stage 2: Install dependencies and built package to a virtual environment. This
# means we only have the main dependencies installed at this stage and the package
# itself.
FROM python:3.12-slim AS deps

RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV POETRY_HOME="/opt/poetry"
ENV POETRY_VERSION=1.8.3
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY --from=builder /app/pyproject.toml ./
COPY --from=builder /app/poetry.lock ./

RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root

COPY --from=builder /app/dist/ ./dist/
RUN poetry run pip install dist/*.whl



# Stage 3: Minimal runtime image
FROM python:3.12-slim AS runtime

# Security: run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
VOLUME /app/.config/gcloud

# Copy only what's needed from previous stages
COPY --from=deps /app/.venv/lib/python3.12/site-packages/ /usr/local/lib/python3.12/site-packages/

# We also need any assets or static files that the application uses.
COPY ./assets/ ./assets/

USER appuser

# Entrypoint runs the main data download script
CMD ["python", "-m", "pm25ml.run.download_results"]
