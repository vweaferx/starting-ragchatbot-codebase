#!/usr/bin/env bash
# Run all frontend code quality checks.
# Usage: ./scripts/check-frontend.sh [--fix]
set -euo pipefail

FRONTEND_DIR="$(cd "$(dirname "$0")/../frontend" && pwd)"
FIX=false

for arg in "$@"; do
    if [[ "$arg" == "--fix" ]]; then
        FIX=true
    fi
done

echo "==> Installing frontend dependencies..."
npm --prefix "$FRONTEND_DIR" install --silent

if $FIX; then
    echo "==> Formatting with Prettier..."
    npm --prefix "$FRONTEND_DIR" run format

    echo "==> Fixing lint issues with ESLint..."
    npm --prefix "$FRONTEND_DIR" run lint:fix
else
    echo "==> Checking formatting with Prettier..."
    npm --prefix "$FRONTEND_DIR" run format:check

    echo "==> Linting with ESLint..."
    npm --prefix "$FRONTEND_DIR" run lint
fi

echo ""
echo "All frontend quality checks passed."
