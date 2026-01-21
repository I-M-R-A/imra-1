#!/usr/bin/env bash

set -euo pipefail

PYTHONPATH=src poetry run pytest "$@"
