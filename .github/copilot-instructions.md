# Copilot instructions for Roars_valgdata

Purpose
- Guidance for Copilot sessions working with this repository (data analysis notebook + CSV).

Build, test, and lint commands
- No build, test, or lint configuration is present in this repository.
- To open and run the analysis notebook:
  - Install dependencies (example): python -m venv .venv && . .venv/bin/activate && pip install pandas jupyter
  - Run the notebook interactively: jupyter lab . or jupyter notebook analyse_data.ipynb
- There are no automated tests; "running a single test" is not applicable.

High-level architecture
- Single-analysis repo: primary code is a Jupyter notebook: `analyse_data.ipynb`.
- Data: `Valgresultat 2025👍 (Svar) - Skjemasvar 1.csv` (CSV in repo root). The notebook reads/derives all analysis from this CSV.
- Repo contains only data + notebook + LICENSE; there are no modules, packages, or CI configs to inspect.

Key conventions and notes for Copilot
- Treat the notebook as the canonical analysis flow. When asked to modify analysis, prefer producing a short, standalone Python script (e.g., scripts/run_analysis.py) or a new notebook cell to preserve reproducibility.
- Filenames include non-ASCII characters (emoji and spaces). Use quotes or escaped paths when running commands (or normalize filenames if adding automation).
- No requirements.txt or environment.yml present. If adding reproducible env files, place them at repo root (requirements.txt or environment.yml) and reference them in this file.
- Keep changes minimal and avoid converting the notebook to scripts without user confirmation; suggest creating a script alongside the notebook instead.

Other AI assistant configs
- Searched common AI config filenames; none were found in the repository root.

What was created
- Added .github/copilot-instructions.md containing the above guidance.

If you want changes or additional coverage (for example, suggested requirements, an example run script, or CI/test scaffolding), say which area to expand.
