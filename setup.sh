#!/bin/bash
# Repo setup via Poetry.
#
# 1. Installs Poetry (via the official installer) if not already on PATH.
# 2. Configures Poetry to create the virtualenv inside the project (.venv/).
# 3. Installs pinned dependencies from pyproject.toml / poetry.lock.
#
# After this completes, activate the environment with:
#     source .venv/bin/activate
#
# Environment overrides:
#   POETRY_HOME   — installation prefix for Poetry itself (default: $HOME/.local)

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

POETRY_HOME="${POETRY_HOME:-$HOME/.local}"

ensure_poetry_on_path() {
    if command -v poetry >/dev/null 2>&1; then
        return 0
    fi
    for candidate in "$POETRY_HOME/bin" "$HOME/.local/bin" "$HOME/.poetry/bin"; do
        if [[ -x "$candidate/poetry" ]]; then
            export PATH="$candidate:$PATH"
            return 0
        fi
    done
    return 1
}

if ! ensure_poetry_on_path; then
    echo "Installing Poetry into $POETRY_HOME ..."
    if ! command -v curl >/dev/null 2>&1; then
        echo "ERROR: curl is required to install Poetry but is not on PATH." >&2
        exit 1
    fi
    if ! curl -sSL https://install.python-poetry.org | POETRY_HOME="$POETRY_HOME" python3 -; then
        echo "ERROR: Poetry installation failed." >&2
        echo "If you are offline or behind a proxy, install Poetry manually and re-run." >&2
        exit 1
    fi
    ensure_poetry_on_path || {
        echo "ERROR: Poetry installed but still not on PATH. Try: export PATH=\"$POETRY_HOME/bin:\$PATH\"" >&2
        exit 1
    }
fi

echo "Using Poetry: $(poetry --version) at $(command -v poetry)"

# Keep the virtualenv in-project so it matches the .gitignore entry (.venv).
poetry config virtualenvs.in-project true --local

echo "Installing dependencies from pyproject.toml ..."
poetry install --no-root

cat <<MSG

Setup complete. Activate the environment with:
    source .venv/bin/activate

Then run the smoke tests:
    bash dev/tests/run_all.sh 01

Or run any training command directly:
    python Training/mingpt/run.py inequality_finetune ...
MSG
