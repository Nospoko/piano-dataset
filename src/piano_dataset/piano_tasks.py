import importlib.resources as pkg_resources

import yaml
import pandas as pd

from piano_dataset.piano_task import (
    PianoTask,
    PromptTaskType,
    TargetPromptSplit,
    ParametricPianoTask,
    ParametricTargetPromptSplit,
)


class PianoTaskManager:
    def __init__(self):
        self._tasks = {}

    def register_task(self, task_cls: PianoTask):
        if not hasattr(task_cls, "name") or not task_cls.name:
            raise ValueError("Task class must have a 'name' attribute.")

        if task_cls.name in self._tasks:
            raise ValueError(f"Task '{task_cls.name}' is already registered.")
        self._tasks[task_cls.name] = task_cls

    def get_task(self, task_name: str) -> PianoTask:
        task_cls = self._tasks.get(task_name)
        task = task_cls()
        return task

    def list_tasks(self) -> list[str]:
        return list(self._tasks.keys())


class AboveMedianPrediction(PianoTask):
    name = "above_median_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<LOW_FROM_MEDIAN>"
    target_token = "<HIGH_FROM_MEDIAN>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        median = notes_df.pitch.median()
        source_df = notes_df[notes_df.pitch < median].reset_index(drop=True)
        target_df = notes_df[notes_df.pitch >= median].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class AboveLowQuartilePrediction(PianoTask):
    name = "above_low_quartile_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<BELOW_LOW_QUARTILE>"
    target_token = "<ABOVE_LOW_QUARTILE>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        q1 = notes_df.pitch.quantile(0.25)
        source_df = notes_df[notes_df.pitch < q1].reset_index(drop=True)
        target_df = notes_df[notes_df.pitch >= q1].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class AboveHighQuartilePrediction(PianoTask):
    name = "above_high_quartile_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<BELOW_HIGH_QUARTILE>"
    target_token = "<ABOVE_HIGH_QUARTILE>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        q3 = notes_df.pitch.quantile(0.75)
        source_df = notes_df[notes_df.pitch < q3].reset_index(drop=True)
        target_df = notes_df[notes_df.pitch >= q3].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class BelowLowQuartilePrediction(PianoTask):
    name = "below_low_quartile_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<ABOVE_LOW_QUARTILE>"
    target_token = "<BELOW_LOW_QUARTILE>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        q1 = notes_df.pitch.quantile(0.25)
        source_df = notes_df[notes_df.pitch >= q1].reset_index(drop=True)
        target_df = notes_df[notes_df.pitch < q1].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class BelowHighQuartilePrediction(PianoTask):
    name = "below_high_quartile_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<ABOVE_HIGH_QUARTILE>"
    target_token = "<BELOW_HIGH_QUARTILE>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        q3 = notes_df.pitch.quantile(0.75)
        source_df = notes_df[notes_df.pitch >= q3].reset_index(drop=True)
        target_df = notes_df[notes_df.pitch < q3].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class BelowMedianPrediction(PianoTask):
    name = "below_median_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<HIGH_FROM_MEDIAN>"
    target_token = "<LOW_FROM_MEDIAN>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        median = notes_df.pitch.median()
        source_df = notes_df[notes_df.pitch >= median].reset_index(drop=True)
        target_df = notes_df[notes_df.pitch < median].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class MiddleQuartilesPrediction(PianoTask):
    name = "middle_quartile_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<EXTREME_QUARTILES>"
    target_token = "<MIDDLE_QUARTILES>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        q1 = notes_df.pitch.quantile(0.25)
        q3 = notes_df.pitch.quantile(0.75)

        ids = (notes_df.pitch < q1) | (notes_df.pitch >= q3)
        source_df = notes_df[ids].reset_index(drop=True)
        target_df = notes_df[~ids].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class ExtremeQuartilesPrediction(PianoTask):
    name = "extreme_quartile_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<MIDDLE_QUARTILES>"
    target_token = "<EXTREME_QUARTILES>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        q1 = notes_df.pitch.quantile(0.25)
        q3 = notes_df.pitch.quantile(0.75)
        ids = (notes_df.pitch < q1) | (notes_df.pitch >= q3)
        target_df = notes_df[ids]
        source_df = notes_df[~ids]

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class FastNotesPrediction(PianoTask):
    name = "fast_notes_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<SLOW_NOTES>"
    target_token = "<FAST_NOTES>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        ids = notes_df.duration < 0.2
        target_df = notes_df[ids]
        source_df = notes_df[~ids]

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class TopLinePrediction(PianoTask):
    name = "top_line_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<BELOW_TOP_LINE>"
    target_token = "<TOP_LINE>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        start_time = notes_df.start.min()
        end_time = notes_df.start.max()

        window_size = 0.2
        top_line_idxs = []
        while start_time <= end_time:
            # Get notes that were ringing during this time interval
            ids = (notes_df["start"] <= start_time + window_size) & (notes_df["end"] > start_time)
            if ids.any():
                idx = notes_df[ids].pitch.idxmax()
                top_line_idxs.append(idx)

            start_time += window_size

        ids = notes_df.index.isin(top_line_idxs)
        target_df = notes_df[ids]
        source_df = notes_df[~ids]

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class LowLinePrediction(PianoTask):
    name = "low_line_prediction"
    type = PromptTaskType.COMBINE
    source_token = "<ABOVE_LOW_LINE>"
    target_token = "<LOW_LINE>"

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        start_time = notes_df.start.min()
        end_time = notes_df.start.max()

        window_size = 0.2
        top_line_idxs = []
        while start_time <= end_time:
            # Get notes that were ringing during this time interval
            ids = (notes_df["start"] <= start_time + window_size) & (notes_df["end"] > start_time)
            if ids.any():
                idx = notes_df[ids].pitch.idxmin()
                top_line_idxs.append(idx)

            start_time += window_size

        ids = notes_df.index.isin(top_line_idxs)
        target_df = notes_df[ids]
        source_df = notes_df[~ids]

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            source_token=self.source_token,
            target_token=self.target_token,
        )
        return target_split


class ParametricTopLineMasking(ParametricPianoTask):
    name = "parametric_top_line_masking"
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

    def prompt_target_split(self, notes_df: pd.DataFrame) -> ParametricTargetPromptSplit:
        source_df = notes_df.copy()
        target_notes = []
        for it in range(self.n_repetitions):
            top_line_idxs = self._find_top_line_notes(source_df)

            ids = source_df.index.isin(top_line_idxs)

            target_df = source_df[ids]
            target_notes.append(target_df)

            source_df = source_df[~ids]

        target_df = pd.concat(target_notes, axis=0)

        target_split = ParametricTargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class ParametricHighNotesMasking(ParametricPianoTask):
    name = "parametric_high_notes_masking"
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

    def prompt_target_split(self, notes_df: pd.DataFrame) -> ParametricTargetPromptSplit:
        target_df = notes_df.nlargest(self.n_notes, "pitch")

        ids = notes_df.index.isin(target_df.index)
        source_df = notes_df[~ids].reset_index(drop=True)

        source_df = source_df.sort_values(by="start", ignore_index=True)
        target_df = target_df.sort_values(by="start", ignore_index=True)
        target_split = ParametricTargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class ParametricShortNotesMasking(ParametricPianoTask):
    name = "parametric_short_notes_masking"
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

    def prompt_target_split(self, notes_df: pd.DataFrame) -> ParametricTargetPromptSplit:
        target_df = notes_df.nsmallest(self.n_notes, "duration")

        ids = notes_df.index.isin(target_df.index)
        source_df = notes_df[~ids].reset_index(drop=True)

        source_df = source_df.sort_values(by="start", ignore_index=True)
        target_df = target_df.sort_values(by="start", ignore_index=True)
        target_split = ParametricTargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class ParametricStrideMasking(ParametricPianoTask):
    name = "parametric_stride_notes_masking"
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

    def prompt_target_split(self, notes_df: pd.DataFrame) -> ParametricTargetPromptSplit:
        # It's an arbitrary choice, but for now let the target
        # always start from the second note
        # My version of stride: retain N rows, skip N rows
        N = self.stride
        target_notes = [notes_df.iloc[it : it + N] for it in range(N, len(notes_df), 2 * N)]
        target_df = pd.concat(target_notes)

        source_notes = [notes_df.iloc[it : it + N] for it in range(0, len(notes_df), 2 * N)]
        source_df = pd.concat(source_notes)

        target_split = ParametricTargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class ParametricTaskManager:
    _task_registry = {task_class.__name__: task_class for task_class in ParametricPianoTask.__subclasses__()}

    def __init__(self, tasks_config: dict):
        self.tasks_config = tasks_config
        self.tasks = {}

        for task_config in tasks_config:
            piano_task = self.create_task(
                class_name=task_config["class"],
                params=task_config.get("params", {}),
            )
            self.tasks[piano_task.task_name] = piano_task

    def __rich_repr__(self):
        yield "n_tasks", len(self.tasks)
        yield "tasks", self.list_task_names()

    def list_task_names(self) -> list[str]:
        task_names = [task_name for task_name in self.tasks]
        return task_names

    def get_task(self, task_name) -> ParametricPianoTask:
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
    ) -> ParametricPianoTask:
        if class_name not in self._task_registry:
            raise ValueError(f"Unknown task class: {class_name}")

        piano_task = self._task_registry[class_name](
            **params,
        )
        return piano_task

    @classmethod
    def load_default(cls) -> "ParametricTaskManager":
        with pkg_resources.open_text("configs", "tasks-default.yaml") as f:
            tasks_config = yaml.safe_load(f)

        this = cls(tasks_config)
        return this
