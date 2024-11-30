# PIANO Dataset

Demo: https://huggingface.co/spaces/epr-labs/PIANO-Dataset

Usage:

```python
from piano_dataset import PianoTasks
from datasets import load_dataset
from fortepyan import MidiPiece


dataset = load_dataset("epr-labs/maestro-sustain-v2", split="test")
piece = MidiPiece.from_huggingface(dataset[33])

PianoTasks.list_tasks()

# [
#     'above_median_prediction',
#     'above_low_quartile_prediction',
#     'above_high_quartile_prediction',
#     'below_low_quartile_prediction',
#     'below_high_quartile_prediction',
#     'below_median_prediction',
#     'middle_quartile_prediction',
#     'extreme_quartile_prediction'
#     ...
# ]

piano_task = PianoTasks.get_task(task_name='extreme_quartile_prediction')
target_prompt = piano_task.prompt_target_split(notes_df=piece.df)

target_prompt
# TargetPromptSplit(
#     'TargetPromptSplit',
#     source_notes=1588,
#     target_notes=1538,
#     source_token='<MIDDLE_QUARTILES>',
#     target_token='<EXTREME_QUARTILES>'
# )
```

## Development

```sh
pip install -e .[dev]
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
