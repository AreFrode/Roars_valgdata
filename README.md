# Roars_valgdata

Enkel analyse av roar sin valgdata (small, single-notebook analysis).

[![CI](https://github.com/your-username/your-repo/actions/workflows/ci.yml/badge.svg)](https://github.com/your-username/your-repo/actions/workflows/ci.yml)

This repository contains:
- analyse_data.ipynb — primary Jupyter notebook with the data analysis flow.
- "Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv" — raw CSV data (note: filename contains spaces and an emoji).
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
  "Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv"
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

How to create a GPG key and add secrets (recommended)
1) Generate a passphrase-protected GPG key locally (example):

   gpg --batch --gen-key <<'EOF'
   Key-Type: RSA
   Key-Length: 4096
   Subkey-Type: RSA
   Subkey-Length: 4096
   Name-Real: Your Name
   Name-Email: you@example.com
   Expire-Date: 0
   EOF

   This will interactively create a key; omit --batch for interactive prompts. Use a strong passphrase and remember it.

2) Export the private key (ASCII-armored) and copy it into the GPG_PRIVATE_KEY secret:

   gpg --armor --export-secret-keys you@example.com > private.key.asc
   # Then open private.key.asc and copy its full contents into the GPG_PRIVATE_KEY repository secret.

3) Create an API token on PyPI (Account → API tokens) and add as PYPI_API_TOKEN; create a TestPyPI token for TEST_PYPI_API_TOKEN.

4) Once secrets are configured, creating a GitHub release will trigger the workflow which:
   - builds the package
   - imports the GPG key and signs artifacts (if GPG_PRIVATE_KEY/GPG_PASSPHRASE are set)
   - uploads signed artifacts to TestPyPI or PyPI depending on release type

Local publishing (example)
- Build and upload to TestPyPI (locally; requires twine and TEST_PYPI_API_TOKEN):

  python -m build
  python -m pip install --upgrade twine
  twine upload --repository testpypi dist/* -u __token__ -p <YOUR_TEST_PYPI_API_TOKEN>

Security notes
- Do not commit private keys or API tokens into the repository. Use repository secrets for CI and local environment variables for temporary tests.
- Keep the GPG private key secure and rotate PyPI tokens if compromised.

If anything above should be clarified further (more examples, screenshots, or step-by-step token creation), say which part to expand.

If you ever need anything installed, just ask me to install it: provide the installation command(s) you want run and paste the command output here so I can continue (or run them locally and paste results).
