#!/usr/bin/env bash
# PTCG AI Battle — Kaggle Submission Packager
# Usage: ./submit.sh [command]
#
# Commands:
#   pack      Create submission tarball (default)
#   check     Run smoke test without packaging
#   clean     Remove build artifacts

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/.submission_build"
SUBMISSION_NAME="ptcg-agent-submission"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }

# ── Files to include in submission ──
# These are the files Kaggle needs. Everything else is dev-only.
SUBMIT_FILES=(
    main.py
    deck.csv
    model.pt
    cg/__init__.py
    cg/api.py
    cg/game.py
    cg/sim.py
    cg/utils.py
    cg/cg.dll
    cg/libcg.so
    agent/__init__.py
    agent/main.py
    agent/policy.py
    agent/search.py
    agent/evaluate.py
    agent/features.py
    agent/network.py
    agent/deck.py
)

# ── Pack command ──
cmd_pack() {
    info "Building submission package..."

    rm -rf "$BUILD_DIR"
    mkdir -p "$BUILD_DIR/$SUBMISSION_NAME/cg" "$BUILD_DIR/$SUBMISSION_NAME/agent"

    # Copy submission files
    for f in "${SUBMIT_FILES[@]}"; do
        src="$PROJECT_ROOT/$f"
        dst="$BUILD_DIR/$SUBMISSION_NAME/$f"
        if [ ! -f "$src" ]; then
            error "Missing required file: $f"
        fi
        cp "$src" "$dst"
    done

    # Remove any __pycache__ from build
    find "$BUILD_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

    # Verify key files
    verify_deck
    verify_model
    verify_imports

    # Create tarball with files at archive root (not in a subdirectory)
    find "$BUILD_DIR/$SUBMISSION_NAME" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find "$BUILD_DIR/$SUBMISSION_NAME" -name '*.pyc' -delete 2>/dev/null || true

    cd "$BUILD_DIR/$SUBMISSION_NAME"
    tar czf "$PROJECT_ROOT/$SUBMISSION_NAME.tar.gz" \
        main.py deck.csv model.pt \
        cg/ agent/
    rm -rf "$BUILD_DIR/$SUBMISSION_NAME"

    SIZE=$(du -h "$PROJECT_ROOT/$SUBMISSION_NAME.tar.gz" | cut -f1)
    info "Package created: $SUBMISSION_NAME.tar.gz ($SIZE)"
    info ""
    info "Contents:"
    tar tzf "$PROJECT_ROOT/$SUBMISSION_NAME.tar.gz" | head -20
}

# ── Check command — run smoke test ──
cmd_check() {
    info "Running smoke test..."
    python3 "$PROJECT_ROOT/smoke_test.py"
}

# ── Clean command ──
cmd_clean() {
    rm -rf "$BUILD_DIR"
    rm -f "$PROJECT_ROOT/$SUBMISSION_NAME.tar.gz"
    rm -rf "$PROJECT_ROOT/si_training/gpu_test"
    info "Cleaned build artifacts"
}

# ── Verify deck ──
verify_deck() {
    local deck_file="$BUILD_DIR/$SUBMISSION_NAME/deck.csv"
    local count
    count=$(wc -l < "$deck_file" | tr -d ' ')
    if [ "$count" -ne 60 ]; then
        error "deck.csv must have exactly 60 lines, got $count"
    fi
    info "deck.csv: 60 cards ✓"
}

# ── Verify model ──
verify_model() {
    local model_file="$BUILD_DIR/$SUBMISSION_NAME/model.pt"
    if [ ! -f "$model_file" ]; then
        warn "model.pt not found — agent will use heuristic-only mode"
    else
        local size
        size=$(du -h "$model_file" | cut -f1)
        info "model.pt: $size ✓"
    fi
}

# ── Verify imports ──
verify_imports() {
    cd "$BUILD_DIR/$SUBMISSION_NAME"
    python3 -c "
import sys, ast
sys.path.insert(0, '.')

# Parse check only — don't import (avoids __pycache__)
with open('main.py') as f:
    ast.parse(f.read())
print('main.py: syntax OK')

with open('deck.csv') as f:
    lines = [l.strip() for l in f if l.strip()]
assert len(lines) == 60
print('deck.csv: 60 cards OK')
" || error "File verification failed"

    # Remove any __pycache__ before tarring
    find "$BUILD_DIR" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    find "$BUILD_DIR" -name '*.pyc' -delete 2>/dev/null || true

    cd "$PROJECT_ROOT"
    info "Import test: ✓"
}

# ── Main ──
COMMAND="${1:-pack}"
case "$COMMAND" in
    pack)  cmd_pack  ;;
    check) cmd_check ;;
    clean) cmd_clean ;;
    *)     echo "Usage: $0 {pack|check|clean}"; exit 1 ;;
esac