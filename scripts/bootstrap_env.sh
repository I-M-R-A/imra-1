#!/usr/bin/env bash

set -euo pipefail

poetry --version >/dev/null 2>&1 || {
  echo "Poetry is required. Install via https://python-poetry.org/docs/" >&2
  exit 1
}

poetry install
poetry run pre-commit install || true
