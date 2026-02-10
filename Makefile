VENV=.venv
PY=python3
PIP=pip

venv:
	$(PY) -m venv $(VENV)

install: venv
	. $(VENV)/bin/activate && python -m pip install --upgrade pip setuptools wheel
	. $(VENV)/bin/activate && pip install -r requirements.txt
	. $(VENV)/bin/activate && pip install -r requirements-dev.txt

test:
	. $(VENV)/bin/activate && PYTHONPATH=src $(VENV)/bin/python -m pytest -q

format:
	. $(VENV)/bin/activate && black src scripts tests

lint:
	. $(VENV)/bin/activate && mypy src || true

notebook:
	. $(VENV)/bin/activate && PYTHONPATH=. $(VENV)/bin/python -m nbconvert --to notebook --execute analyse_data.ipynb --ExecutePreprocessor.timeout=600

run:
	./scripts/run_analysis.py

precommit:
	. $(VENV)/bin/activate && pre-commit run --all-files || true

docs:
	. $(VENV)/bin/activate && pip install --upgrade pip
	. $(VENV)/bin/activate && pip install mkdocs mkdocs-material
	. $(VENV)/bin/activate && mkdocs gh-deploy --force
