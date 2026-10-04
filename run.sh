#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# IntelliPDF launcher: starts the FastAPI backend and the Streamlit UI.
#
#   ./run.sh            start both (API :8000, UI :8501)
#   ./run.sh api        backend only
#   ./run.sh ui         Streamlit only
#   ./run.sh setup      create .venv and install dependencies
#   ./run.sh test       run the pytest suite
#   ./run.sh eval       run the SQuAD 2.0 evaluation
#   ./run.sh check      end-to-end smoke test against a running server
#
# Windows (Git Bash) and Linux/macOS are both supported.
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

cd "$(dirname "$0")"

API_PORT="${API_PORT:-8000}"
UI_PORT="${UI_PORT:-8501}"
VENV_DIR="${VENV_DIR:-.venv}"
export PYTHONIOENCODING=utf-8

# ── helpers ──────────────────────────────────────────────────────────────
find_python() {
    for candidate in python3 python py; do
        if command -v "$candidate" >/dev/null 2>&1; then
            echo "$candidate"
            return 0
        fi
    done
    echo "ERROR: python not found. Install Python 3.10+ from https://python.org" >&2
    exit 1
}

PYTHON="$(find_python)"

if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating virtual environment in $VENV_DIR"
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Resolve the venv interpreter for both POSIX and Windows layouts.
if [ -x "$VENV_DIR/Scripts/python.exe" ]; then
    PY="$VENV_DIR/Scripts/python.exe"
else
    PY="$VENV_DIR/bin/python"
fi

wait_for_api() {
    echo "==> Waiting for the API on http://127.0.0.1:$API_PORT/health"
    for _ in $(seq 1 60); do
        if "$PY" - "$API_PORT" <<'PYCHECK' 2>/dev/null
import socket, sys
with socket.create_connection(("127.0.0.1", int(sys.argv[1])), timeout=1):
    pass
PYCHECK
        then
            return 0
        fi
        sleep 1
    done
    echo "ERROR: the API did not start in 60s. Check the logs above." >&2
    return 1
}

start_api() {
    echo "==> Starting FastAPI on port $API_PORT (docs: http://127.0.0.1:$API_PORT/docs)"
    "$PY" -m uvicorn backend.main:app --reload --port "$API_PORT" &
    API_PID=$!
    wait_for_api
}

start_ui() {
    echo "==> Starting Streamlit on port $UI_PORT"
    "$PY" -m streamlit run frontend/app.py \
        --server.port "$UI_PORT" \
        --server.headless true &
    UI_PID=$!
}

cleanup() {
    echo
    echo "==> Shutting down IntelliPDF"
    [ -n "${API_PID:-}" ] && kill "$API_PID" 2>/dev/null || true
    [ -n "${UI_PID:-}" ] && kill "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

case "${1:-all}" in
    setup)
        "$PY" -m pip install --upgrade pip
        "$PY" -m pip install -r requirements.txt
        [ -f .env ] || cp .env.example .env
        echo "==> Done. Put your GROQ_API_KEY in .env, then run ./run.sh"
        ;;
    api)
        start_api
        wait
        ;;
    ui)
        start_ui
        wait
        ;;
    test)
        "$PY" -m pytest tests -v
        ;;
    eval)
        shift || true
        "$PY" evaluation/squad_eval.py "$@"
        ;;
    check)
        shift || true
        "$PY" tests/e2e_check.py "${1:-sample_features.pdf}"
        ;;
    all)
        [ -f .env ] || cp .env.example .env
        start_api
        start_ui
        cat <<'BANNER'

  IntelliPDF is running
  ─────────────────────────────────────────────
   UI  : http://localhost:8501
   API : http://127.0.0.1:8000/docs
   Log : uvicorn/streamlit output above (Ctrl+C to stop)

BANNER
        wait
        ;;
    *)
        echo "Usage: ./run.sh [setup|api|ui|test|eval|check|all]" >&2
        exit 1
        ;;
esac