# PIANO Dataset

Demo: https://huggingface.co/spaces/epr-labs/PIANO-Dataset

## Development

```sh
pip install -e .[dev]
```

To run internal dashboards during development:

```sh
PYTHONPATH=. streamlit run src/piano_dataset/dashboards/task_review.py
```

## Code style
This repository uses pre-commit hooks with forced python formatting ([black](https://github.com/psf/black),
[flake8](https://flake8.pycqa.org/en/latest/), and [isort](https://pycqa.github.io/isort/)):

```sh
pip install pre-commit
pre-commit install
```

Whenever you execute `git commit` the files altered / added within the commit will be checked and corrected.
`black` and `isort` can modify files locally - if that happens you have to `git add` them again.
You might also be prompted to introduce some fixes manually.

To run the hooks against all files without running git commit:

```sh
pre-commit run --all-files
```
