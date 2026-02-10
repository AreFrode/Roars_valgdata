# Contributing

Thanks for your interest in contributing to Roars_valgdata. Small, focused contributions are welcome.

How to contribute

- Fork the repository and create a feature branch.
- Run the test suite and linters locally (see Makefile):

  make venv && . .venv/bin/activate
  make install
  make test

- Run pre-commit hooks and formatters before committing:

  make format
  pre-commit run --all-files

- Open a Pull Request with a clear description and link to related issues.

Pull request guidance

- Keep changes small and focused.
- Add tests for new behavior where applicable.
- Reference the issue number in the PR description.
