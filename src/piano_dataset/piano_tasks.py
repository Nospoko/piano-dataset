import importlib.resources as pkg_resources

import yaml
import numpy as np
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


class ChordMasking(PianoTask):
    name = "chord_masking"
    type = PromptTaskType.COMBINE
    task_token = "<CHORD_MASK>"

    def __init__(self, n_notes_threshold: int = 3):
        self.n_notes_threshold = n_notes_threshold

    @property
    def task_name(self) -> str:
        name = f"{self.name}-x{self.n_notes_threshold}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        mask = notes_df.start.diff() < 0.015

        # Identify start of new groups
        groups = (mask != mask.shift()).cumsum()

        # Count the size of each group
        group_sizes = groups.map(groups.value_counts())

        # Time differences from the first note of the chord
        ids = (mask) & (group_sizes >= self.n_notes_threshold - 1)

        # And include the first note as well
        # NOTE Actually let's NOT include the first not, and
        # expect the models to build chords on top of existing notes
        # NOTE It's going to build on top of the first note, not the lowest
        # ids = ids | ids.shift(-1, fill_value=False)

        source_df = notes_df[~ids].reset_index(drop=True)
        target_df = notes_df[ids].reset_index(drop=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class RandomMasking(PianoTask):
    name = "random_masking"
    type = PromptTaskType.COMBINE
    task_token = "<RANDOM_MASK>"

    def __init__(self, mask_percentage: int = 50):
        self.mask_percentage = mask_percentage

    @property
    def masking_fraction(self) -> float:
        return self.mask_percentage / 100

    @property
    def task_name(self) -> str:
        name = f"{self.name}-x{self.mask_percentage}"
        return name

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        target_df = notes_df.sample(frac=self.masking_fraction)
        source_df = notes_df.drop(target_df.index)

        source_df = source_df.reset_index(drop=True)
        target_df = target_df.sort_values("start", ignore_index=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class LinearRandomMasking(PianoTask):
    name = "linear_random_masking"
    type = PromptTaskType.COMBINE
    task_token = "<LINEAR_RANDOM_MASK>"

    def __init__(self, mask_percentage: int = 50):
        self.mask_percentage = mask_percentage

    @property
    def task_name(self) -> str:
        name = f"{self.name}-x{self.mask_percentage}"
        return name

    @property
    def masking_fraction(self) -> float:
        return self.mask_percentage / 100

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        time_min = notes_df.start.min()
        time_max = notes_df.start.max()

        pitch_start = notes_df[:10].sample().iloc[0].pitch
        pitch_finish = notes_df[-10:].sample().iloc[0].pitch

        dx = time_max - time_min
        dy = pitch_finish - pitch_start

        num = np.abs(
            dy * notes_df["start"] - dx * notes_df["pitch"] + time_max * pitch_start - pitch_finish * time_min,
        )
        den = np.sqrt(dx**2 + dy**2)
        distances = num / den

        r = 2 + np.random.randint(7)
        ids = distances <= r

        target_df = notes_df[ids].sample(frac=self.masking_fraction)
        source_df = notes_df.drop(target_df.index)

        source_df = source_df.reset_index(drop=True)
        target_df = target_df.sort_values("start", ignore_index=True)

        target_split = TargetPromptSplit(
            source_df=source_df,
            target_df=target_df,
            prefix_tokens=self.prefix_tokens,
        )
        return target_split


class BandRandomMasking(PianoTask):
    name = "band_random_masking"
    type = PromptTaskType.COMBINE
    task_token = "<BAND_RANDOM_MASK>"

    def __init__(self, mask_percentage: int = 50):
        self.mask_percentage = mask_percentage

    @property
    def task_name(self) -> str:
        name = f"{self.name}-x{self.mask_percentage}"
        return name

    @property
    def masking_fraction(self) -> float:
        return self.mask_percentage / 100

    @property
    def prefix_tokens(self) -> list[str]:
        prefix_tokens = [
            self.task_token,
        ]
        return prefix_tokens

    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        time_min = notes_df.start.min()
        time_max = notes_df.start.max()

        pitch_start = notes_df[:10].sample().iloc[0].pitch
        pitch_finish = notes_df[-10:].sample().iloc[0].pitch

        r0 = 1 + np.random.randint(10)
        r1 = 1 + np.random.randint(10)

        # Slopes of the bounding lines
        m_upper = (pitch_finish + r1 - (pitch_start + r0)) / (time_max - time_min)
        m_lower = (pitch_finish - r1 - (pitch_start - r0)) / (time_max - time_min)

        # Intercepts
        b_upper = (pitch_start + r0) - m_upper * time_min
        b_lower = (pitch_start - r0) - m_lower * time_min

        # For each point, check if pitch lies between the two lines at that time
        y_upper = m_upper * notes_df["start"] + b_upper
        y_lower = m_lower * notes_df["start"] + b_lower

        mask = (notes_df["pitch"] <= y_upper) & (notes_df["pitch"] >= y_lower)

        target_df = notes_df[mask].sample(frac=self.masking_fraction)
        source_df = notes_df.drop(target_df.index)

        source_df = source_df.reset_index(drop=True)
        target_df = target_df.sort_values("start", ignore_index=True)

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
