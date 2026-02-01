#!/usr/bin/env bash
set -euo pipefail

VERSION=$(grep -m1 '^version' Cargo.toml | sed 's/.*"\(.*\)"/\1/')
TAG="v${VERSION}"

echo "Releasing ${TAG}"

if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: working tree is not clean" >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
if [ "$BRANCH" != "main" ]; then
    echo "ERROR: not on main branch (on '${BRANCH}')" >&2
    exit 1
fi

if git rev-parse "$TAG" >/dev/null 2>&1; then
    echo "ERROR: tag ${TAG} already exists" >&2
    exit 1
fi

echo "Running lint..."
make lint

echo "Running tests..."
make test

echo "Running miri tests..."
make test-miri

echo "Running cargo publish --dry-run..."
cargo publish --dry-run

echo ""
echo "All checks passed. Ready to tag ${TAG} and publish to crates.io."
read -rp "Proceed? [y/N] " confirm
if [[ "$confirm" != [yY] ]]; then
    echo "Aborted." >&2
    exit 1
fi

echo "Creating tag ${TAG}..."
git tag -a "$TAG" -m "Release ${TAG}"

echo "Publishing to crates.io..."
cargo publish

echo "Pushing tag to origin..."
git push origin "$TAG"

echo "Creating GitHub release..."
gh release create "$TAG" --title "$TAG" --generate-notes --draft --verify-tag

echo "Released ${TAG} successfully"
