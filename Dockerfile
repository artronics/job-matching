FROM python:3.10-buster

RUN pip install poetry

WORKDIR /app

COPY models models
COPY jobmatch jobmatch
COPY pyproject.toml .
COPY README.md .

RUN poetry install

ENTRYPOINT ["poetry", "run", "python", "-m", "jobmatch.api"]
