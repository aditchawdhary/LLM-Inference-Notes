GitHub CI Equivalent Command:
pre-commit run --all-files --hook-stage manual
This command will run all hooks that are configured to run in CI (marked with stages: [manual] in the config).

What Runs in CI vs Locally:
Looking at the .pre-commit-config.yaml, here's what runs where:

Local Only (stages: [pre-commit]):
mypy-local - Uses your local Python installation
CI Only (stages: [manual]):
markdownlint - Markdown linting
mypy-3.9, mypy-3.10, mypy-3.11, mypy-3.12 - Type checking for multiple Python versions
Various other CI-specific checks
Both Local and CI (default stages):
yapf - Code formatting
ruff - Linting and fixing
ruff-format - Additional formatting
isort - Import sorting
clang-format - C++ formatting
typos - Spell checking
actionlint - GitHub Actions linting
All the custom local hooks
Alternative: Run Specific CI Hooks:
You can also run individual CI hooks:



# Run mypy for specific Python versions (like CI does)
pre-commit run mypy-3.11 --all-files
pre-commit run mypy-3.12 --all-files

# Run markdown linting (like CI does)
pre-commit run markdownlint --all-files

```
pip install pre-commit yapf ruff isort mypy
pre-commit run --all-files --hook-stage manual
```
