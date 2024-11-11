import pandas as pd

from piano_dataset.piano_task import PianoTask, PromptTaskType, TargetPromptSplit


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
