import json

import streamlit as st
from datasets import Dataset, load_dataset


def select_part_dataset(
    midi_dataset: Dataset,
    key: str = "0",
) -> Dataset:
    """
    Allows the user to select a part of the dataset based on composer and title.

    Parameters:
        midi_dataset (Dataset): The MIDI dataset to select from.

    Returns:
        Dataset: The selected part of the dataset.
    """
    source_df = midi_dataset.to_pandas()
    source_df["source"] = source_df["source"].map(lambda source: json.loads(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        label="Select composer",
        options=composers,
        index=8,
        key=f"composer_{key}",
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        label="Select title",
        options=piece_titles,
        key=f"title_{key}",
    )

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


def dataset_configuration(key: str = "0") -> Dataset:
    col1, col2 = st.columns(2)
    with col1:
        dataset_path = st.text_input(
            label="Dataset Path",
            value="epr-labs/maestro-sustain-v2",
            help="Enter the path to the dataset",
            key=f"d_path_{key}",
        )
    with col2:
        dataset_split = st.selectbox(
            label="Dataset Split",
            options=["validation", "train", "test"],
            help="Choose the dataset split to use",
            key=f"split_{key}",
        )

    dataset = load_hf_dataset(
        dataset_path=dataset_path,
        dataset_split=dataset_split,
    )
    dataset = select_part_dataset(
        midi_dataset=dataset,
        key=key,
    )

    return dataset


@st.cache_data
def load_hf_dataset(
    dataset_path: str,
    dataset_split: str,
) -> Dataset:
    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    return dataset
