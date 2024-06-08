FROM tensorflow/tensorflow:latest-gpu

ENV POETRY_VERSION=1.7.1 \
  POETRY_HOME="/usr/local" \
  POETRY_CACHE_DIR="/var/cache/pypoetry" \
  POETRY_NO_INTERACTION=1 \
  POETRY_VIRTUALENVS_CREATE=false \
  PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 \
  PIP_NO_CACHE_DIR=off \
  PIP_DISABLE_PIP_VERSION_CHECK=on \
  PIP_DEFAULT_TIMEOUT=100

RUN apt-get update &&  \
  apt-get install -y --no-install-recommends \
  curl \
  build-essential \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH"

ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g $GROUP_ID appuser && \
  useradd -m -u $USER_ID -g appuser appuser

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --no-interaction --no-ansi

COPY . .
RUN chown -R appuser:appuser /app

USER appuser

CMD ["poetry", "run", "python", "kaggle_titanic/titanic__tensorflow_2.py"]
