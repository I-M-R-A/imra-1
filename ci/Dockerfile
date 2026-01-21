FROM python:3.11-slim

RUN pip install --no-cache-dir poetry==1.7.1
WORKDIR /workspace
COPY pyproject.toml README.md ./
RUN poetry config virtualenvs.create false && poetry install --no-root
COPY . .
CMD ["pytest"]
