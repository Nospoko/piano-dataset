from enum import Enum
from abc import ABC, abstractmethod
from typing import Union, NamedTuple

import pandas as pd


class PromptTaskType(Enum):
    """
    Defines the type of operation to perform between prompt and target data.

    - COMBINE: Merges the prompt and target DataFrames, typically for tasks
        you expect the model to add notes to the input
    - REPLACE: Model output will be treated as a separte musical generation,
        ignoring the prompt.
    """

    COMBINE = "combine"
    REPLACE = "replace"


class TargetPromptSplit(NamedTuple):
    source_token: str
    target_token: str
    source_df: pd.DataFrame
    target_df: pd.DataFrame

    def __rich_repr__(self):
        yield "source_notes", self.source_df.shape[0]
        yield "target_notes", self.target_df.shape[0]
        yield "source_token", self.source_token
        yield "target_token", self.target_token


class PianoTask(ABC):
    name: str
    source_token: str
    target_token: str
    type: PromptTaskType

    @abstractmethod
    def prompt_target_split(self, notes_df: pd.DataFrame) -> TargetPromptSplit:
        pass

    def __rich_repr__(self):
        yield "name", self.name
        yield "type", self.type.name
        yield "source_token", self.source_token
        yield "target_token", self.target_token


class ParametricTargetPromptSplit(NamedTuple):
    prefix_tokens: list[str]
    source_df: pd.DataFrame
    target_df: pd.DataFrame

    def __rich_repr__(self):
        yield "source_notes", self.source_df.shape[0]
        yield "target_notes", self.target_df.shape[0]
        yield "prefix_tokens", self.prefix_tokens


class ParametricPianoTask(ABC):
    name: str
    task_token: str
    type: PromptTaskType

    @abstractmethod
    def prompt_target_split(self, notes_df: pd.DataFrame) -> ParametricTargetPromptSplit:
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        pass

    @property
    @abstractmethod
    def prefix_tokens(self) -> list[str]:
        pass

    def task_parameter_token(self, parameter: Union[int, str]) -> str:
        token = f"<PARAMETER_{parameter}>"
        return token

    def __rich_repr__(self):
        yield "taks_name", self.task_name
        yield "type", self.type.name
        yield "prefix_tokens", self.prefix_tokens
