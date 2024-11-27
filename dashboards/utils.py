import yaml
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
    source_df["source"] = source_df["source"].map(lambda source: yaml.safe_load(source))
    source_df["composer"] = [source["composer"] for source in source_df.source]
    source_df["title"] = [source["title"] for source in source_df.source]

    composers = source_df.composer.unique()
    selected_composer = st.selectbox(
        "Select composer",
        options=composers,
        index=3,
        key=f"composer_{key}",
    )

    ids = source_df.composer == selected_composer
    piece_titles = source_df[ids].title.unique()
    selected_title = st.selectbox(
        "Select title",
        options=piece_titles,
        key=f"title_{key}",
    )

    ids = (source_df.composer == selected_composer) & (source_df.title == selected_title)
    part_df = source_df[ids]
    part_dataset = midi_dataset.select(part_df.index.values)

    return part_dataset


def dataset_configuration(key: str = "0"):
    st.header("Dataset Configuration")
    col1, col2 = st.columns(2)
    with col1:
        dataset_path = st.text_input(
            "Dataset Path",
            value="epr-labs/maestro-sustain-v2",
            help="Enter the path to the dataset",
            key=f"d_path_{key}",
        )
    with col2:
        dataset_split = st.selectbox(
            "Dataset Split",
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

    st.success(f"Dataset loaded! Total records: {len(dataset)}")
    return dataset


@st.cache_data
def load_hf_dataset(
    dataset_path: str,
    dataset_split: str,
):
    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
        num_proc=8,
    )
    return dataset
