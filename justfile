# List all available recipes
default:
    just --list

# Check that all programs are installed
[group("installation")]
check-versions:
    uv --version  # https://docs.astral.sh/uv/
    just --version  # https://github.com/casey/just
    direnv --version  # https://direnv.net/

# Allow direnv to load environment variables
[group("installation")]
direnv-allow:
    direnv allow

# Create uv virtual environment
[group("installation")]
create-venv:
    uv sync

# Install pre-commit hooks
[group("installation")]
install-pre-commit:
    uv run pre-commit install

# Setup MLFlow (reminder)
[group("installation")]
reminder-mlflow:
    @echo "\033[1;33mRemember to setup MLFlow!\033[0m"

# Setup environment variables (reminder)
[group("installation")]
reminder-env-vars:
    @echo "\033[1;33mRemember to setup the environment variables by editing the .envrc file!\033[0m"

# Setup repo
[group("installation")]
setup: direnv-allow create-venv install-pre-commit reminder-mlflow reminder-env-vars

# Run pre-commit hooks
[group("linting & formatting")]
pre-commit:
    uv run pre-commit run --all

# Run tests
[group("testing")]
test:
    uv run pytest

# Run tests with coverage
[group("testing")]
test-cov:
    uv run pytest --cov=src --cov-report=html

# Create a new version tag (will trigger publish.yaml workflow)
[group("packaging")]
publish:
    #!/usr/bin/env bash
    # Get last tag from git
    CURRENT_VERSION=$(git describe --tags --abbrev=0)
    echo "Current version: $CURRENT_VERSION"

    # Remove 'v' prefix if it exists
    VERSION_NUMBER=$(echo $CURRENT_VERSION | sed 's/^v//')

    # Split version into major.minor.patch
    IFS='.' read -r MAJOR MINOR PATCH <<< "$VERSION_NUMBER"

    # Increment patch version
    NEW_PATCH=$((PATCH + 1))
    NEW_VERSION="$MAJOR.$MINOR.$NEW_PATCH"

    # Create git tag (always with 'v' prefix)
    NEW_TAG="v$NEW_VERSION"
    echo "New version: $NEW_TAG"

    # Create and push the new tag
    git tag -a "$NEW_TAG" -m "Release version $NEW_VERSION"
    git push origin "$NEW_TAG"

# Print tree of the project (requires installing tree)
[group("tools")]
tree:
    tree -a -I ".venv|.git|.pytest_cache|.coverage|dist|__pycache__|.vscode|.ruff_cache" --dirsfirst

# Run mlflow server
[group("tools")]
mlflow-server:
    mlflow server --backend-store-uri logs/mlflow/mlruns

# Clean logs directory
[group("cleanup")]
clean-logs:
    rm -rf logs/* && touch logs/.gitkeep
