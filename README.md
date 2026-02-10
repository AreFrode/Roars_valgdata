# Roars_valgdata

Enkel analyse av roar sin valgdata (small, single-notebook analysis).

[![CI](https://github.com/your-username/your-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/ci.yml)

This repository contains:
- analyse_data.ipynb — primary Jupyter notebook with the data analysis flow.
- "Valgresultat 2025👍 (Svar) - Skjemasvar.csv" — raw CSV data (note: filename contains spaces and an emoji).
- src/valgdata/ — small Python package extracted for programmatic use.
- scripts/run_analysis.py — CLI wrapper that prints/saves a JSON summary of the CSV.
- tests/ — basic tests for I/O and analysis.
- .github/workflows/ — CI (test workflow) and release workflow for packaging and publishing.

Quickstart — run the analysis locally

1) Create and activate a virtual environment (Unix/macOS):

   python3 -m venv .venv
   . .venv/bin/activate

2) Install dependencies (minimal/dev) and run the CLI example:

   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   # Run the bundled CLI (defaults to the included CSV):
   ./scripts/run_analysis.py
   # Or save JSON summary to a file:
   ./scripts/run_analysis.py -o summary.json

Notes about the CSV path
- The CSV filename contains spaces and an emoji; when using it in scripts or shells, quote the path:
  "Valgresultat 2025👍 (Svar) - Skjemasvar.csv"
- The CLI script defaults to the filename in the repo root; supply -i or --input to override.

Running the notebook
- Start Jupyter Lab or Notebook and open analyse_data.ipynb:

  jupyter lab .
  # or
  jupyter notebook analyse_data.ipynb

Testing
- Tests are in tests/ and are runnable with pytest after installing requirements:

  PYTHONPATH=src pytest -q

- The package and tests were designed to run even in minimal environments: code falls back to stdlib csv and list-of-dict handling when pandas is not present. Installing pandas enables the full DataFrame-powered behavior.

Packaging and local editable install
- Install the package locally for development:

  pip install -e .

- Build source and wheel locally:

  python -m build

Release process (CI)
- A release workflow will build and publish on GitHub release creation. It distinguishes prereleases/RCs (uploaded to TestPyPI) and normal releases (uploaded to PyPI).
- Required repository secrets (add in Settings → Secrets):
  - TEST_PYPI_API_TOKEN — TestPyPI API token (used for prereleases)
  - PYPI_API_TOKEN — PyPI API token (used for normal releases)
  - GPG_PRIVATE_KEY — ASCII-armored private key contents used to sign built artifacts
  - GPG_PASSPHRASE — passphrase for the above key (empty string if key has no passphrase)

---

## Toolkit features (from main branch)

En enkel, men omfattende analyse av valgdata - for moro skyld! 🚀

This toolkit provides comprehensive analysis of election prediction data, including error analysis, bias detection, clustering of prediction patterns, and visualizations.

Key components include data loading, error and bias analysis, clustering, and plotting. See src/ for modules and tests/ for examples.

If anything above should be clarified further (more examples, screenshots, or step-by-step token creation), say which part to expand.
