import importlib.resources as pkg_resources

import yaml
import pandas as pd

from piano_dataset.piano_task import PianoTask, PromptTaskType, TargetPromptSplit


class TopLineMasking(PianoTask):
    name = "top_line_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_TOP_LINE>"

    def __init__(self, n_repetitions: int = 1):
        self.n_repetitions = n_repetitions

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.n_repetitions}"
        return name

    def _find_top_line_notes(self, notes_df: pd.DataFrame) -> list[int]:
        start_time = notes_df.start.min()
        end_time = notes_df.start.max()

        # TODO Move to config
        window_size = 0.2
        top_line_idxs = []
        while start_time <= end_time:
            # Get notes that were ringing during this time interval
            ids = (notes_df["start"] <= start_time + window_size) & (notes_df["end"] > start_time)
            if ids.any():
                idx = notes_df[ids].pitch.idxmax()
                top_line_idxs.append(idx)

            start_time += window_size

        return top_line_idxs

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.n_repetitions),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        source_df = notes_df.copy()
        target_notes = []
        for it in range(self.n_repetitions):
            top_line_idxs = self._find_top_line_notes(source_df)

            ids = source_df.index.isin(top_line_idxs)

            target_df = source_df[ids]
            target_notes.append(target_df)

            source_df = source_df[~ids]

        target_df = pd.concat(target_notes, axis=0)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class BottomLineMasking(PianoTask):
    name = "bottom_line_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_BOTTOM_LINE>"

    def __init__(self, n_repetitions: int = 1):
        self.n_repetitions = n_repetitions

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.n_repetitions}"
        return name

    def _find_bottom_line_notes(self, notes_df: pd.DataFrame) -> list[int]:
        start_time = notes_df.start.min()
        end_time = notes_df.start.max()

        # TODO Move to config
        window_size = 0.2
        bottom_line_idxs = []
        while start_time <= end_time:
            # Get notes that were ringing during this time interval
            ids = (notes_df["start"] <= start_time + window_size) & (notes_df["end"] > start_time)
            if ids.any():
                idx = notes_df[ids].pitch.idxmin()
                bottom_line_idxs.append(idx)

            start_time += window_size

        return bottom_line_idxs

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.n_repetitions),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        source_df = notes_df.copy()
        target_notes = []
        for it in range(self.n_repetitions):
            bottom_line_idxs = self._find_bottom_line_notes(source_df)

            ids = source_df.index.isin(bottom_line_idxs)

            target_df = source_df[ids]
            target_notes.append(target_df)

            source_df = source_df[~ids]

        target_df = pd.concat(target_notes, axis=0)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class RangeMasking(PianoTask):
    name = "range_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_RANGE>"

    def __init__(self, start_sec: int = 0, end_sec: int = 5):
        self.start_sec = start_sec
        self.end_sec = end_sec

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.start_sec}-x{self.end_sec}"
        return name

    def _find_idx_in_range(self, notes_df: pd.DataFrame) -> list[int]:
        # TODO: move to config
        tolerance = 0.05

        ids = (notes_df["start"] >= self.start_sec - tolerance) & (notes_df["start"] <= self.end_sec + tolerance)

        return ids

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.start_sec),
            self.task_parameter_token(self.end_sec)
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        source_df = notes_df.copy()

        range_idx = self._find_idx_in_range(source_df)

        target_df = source_df[range_idx]

        source_df = source_df[~range_idx]

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class HighNotesMasking(PianoTask):
    name = "high_notes_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_HIGH_NOTES>"

    def __init__(self, n_notes: int = 5):
        self.n_notes = n_notes

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.n_notes}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.n_notes),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        target_df = notes_df.nlargest(self.n_notes, "pitch")

        ids = notes_df.index.isin(target_df.index)
        source_df = notes_df[~ids].reset_index(drop=True)

        source_df = source_df.sort_values(by="start", ignore_index=True)
        target_df = target_df.sort_values(by="start", ignore_index=True)
        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class LowNotesMasking(PianoTask):
    name = "low_notes_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_LOW_NOTES>"

    def __init__(self, n_notes: int = 5):
        self.n_notes = n_notes

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.n_notes}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.n_notes),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        target_df = notes_df.nsmallest(self.n_notes, "pitch")

        ids = notes_df.index.isin(target_df.index)
        source_df = notes_df[~ids].reset_index(drop=True)

        source_df = source_df.sort_values(by="start", ignore_index=True)
        target_df = target_df.sort_values(by="start", ignore_index=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )

        return target_split


class ShortNotesMasking(PianoTask):
    name = "short_notes_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_SHORT_NOTES>"

    def __init__(self, n_notes: int = 5):
        self.n_notes = n_notes

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.n_notes}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.n_notes),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        target_df = notes_df.nsmallest(self.n_notes, "duration")

        ids = notes_df.index.isin(target_df.index)
        source_df = notes_df[~ids].reset_index(drop=True)

        source_df = source_df.sort_values(by="start", ignore_index=True)
        target_df = target_df.sort_values(by="start", ignore_index=True)
        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class LongNotesMasking(PianoTask):
    name = "long_notes_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_LONG_NOTES>"

    def __init__(self, n_notes: int = 5):
        self.n_notes = n_notes

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.n_notes}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.n_notes),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        target_df = notes_df.nlargest(self.n_notes, "duration")

        ids = notes_df.index.isin(target_df.index)
        source_df = notes_df[~ids].reset_index(drop=True)

        source_df = source_df.sort_values(by="start", ignore_index=True)
        target_df = target_df.sort_values(by="start", ignore_index=True)
        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class StrideMasking(PianoTask):
    name = "stride_notes_masking"
    type = PromptTaskType.COMBINE
    task_token = "<PARAMETRIC_STRIDE_MASK>"

    def __init__(self, stride: int = 2):
        self.stride = stride

    @property
    def task_name(self) -> str:
        # *x* reads *times*
        name = f"{self.name}-x{self.stride}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
            self.task_parameter_token(self.stride),
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        # It's an arbitrary choice, but for now let the target
        # always start from the second note
        # My version of stride: retain N rows, skip N rows
        N = self.stride
        target_notes = [notes_df.iloc[it : it + N] for it in range(N, len(notes_df), 2 * N)]
        target_df = pd.concat(target_notes)

        source_notes = [notes_df.iloc[it : it + N] for it in range(0, len(notes_df), 2 * N)]
        source_df = pd.concat(source_notes)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class PianoTaskManager:
    _task_registry = {task_class.__name__: task_class for task_class in PianoTask.__subclasses__()}

    def __init__(self, tasks_config: dict):
        self.tasks_config = tasks_config
        self.tasks = {}

        for task_config in tasks_config:
            if "params" in task_config:
                piano_task = self.create_task(
                    class_name=task_config["class"],
                    params=task_config.get("params", {}),
                )
                self.tasks[piano_task.task_name] = piano_task
            elif "param_range" in task_config:
                for params in task_config["param_range"]:
                    piano_task = self.create_task(
                        class_name=task_config["class"],
                        params=params,
                    )
                    self.tasks[piano_task.task_name] = piano_task
            else:
                raise ValueError(f"Invalid PIANO task config! {task_config}")

    def __rich_repr__(self):
        yield "n_tasks", len(self.tasks)
        yield "tasks", self.list_task_names()

    def list_task_names(self) -> list[str]:
        task_names = [task_name for task_name in self.tasks]
        return task_names

    def get_task(self, task_name) -> PianoTask:
        task = self.tasks[task_name]
        return task

    def get_special_tokens(self) -> list[str]:
        special_tokens = []
        for piano_task in self.tasks.values():
            special_tokens += piano_task.prefix_tokens

        special_tokens = list(set(special_tokens))
        return special_tokens

    def create_task(
        self,
        class_name: str,
        params: dict,
    ) -> PianoTask:
        if class_name not in self._task_registry:
            raise ValueError(f"Unknown task class: {class_name}")

        piano_task = self._task_registry[class_name](
            **params,
        )
        return piano_task

    @classmethod
    def load_default(cls) -> "PianoTaskManager":
        with pkg_resources.open_text("configs", "tasks-default.yaml") as f:
            tasks_config = yaml.safe_load(f)

        this = cls(tasks_config)
        return this
