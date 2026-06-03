#!/bin/sh
set -e

MIRROR_URL="https://github.com/raza-shaikh-ai/Rag-systems.git"

if git remote get-url mirror >/dev/null 2>&1; then
    git remote set-url mirror "$MIRROR_URL"
else
    git remote add mirror "$MIRROR_URL"
fi

git config core.hooksPath .githooks

echo "Mirror remote configured and hooks enabled."