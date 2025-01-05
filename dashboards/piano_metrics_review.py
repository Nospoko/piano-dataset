import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from piano_dataset import PianoTasks
from dashboards.utils import dataset_configuration
from piano_metrics.piano_metric import MetricsManager


def main():
    st.write("# Piano Metrics")

    dataset = dataset_configuration()
    record_idx = st.number_input(
        label="Record idx",
        value=0,
        min_value=0,
        max_value=len(dataset) - 1,
    )

    piece = ff.MidiPiece.from_huggingface(dataset[record_idx])

    cols = st.columns(2)
    with cols[0]:
        piece_start = st.number_input(
            label="piece start note",
            value=0,
            min_value=0,
            max_value=piece.size,
        )
    with cols[1]:
        piece_finish = st.number_input(
            label="piece finish note",
            value=128,
            min_value=1,
            max_value=piece.size,
        )
    piece_part = piece[piece_start:piece_finish]

    available_tasks = PianoTasks.list_tasks()
    task_name = st.selectbox(
        label="Select PIANO task",
        options=available_tasks,
    )
    piano_task = PianoTasks.get_task(task_name=task_name)
    piece_split = piano_task.prompt_target_split(piece_part.df)

    source_piece = ff.MidiPiece(
        df=piece_split.source_df.copy(),
    )
    target_piece = ff.MidiPiece(
        df=piece_split.target_df.copy(),
    )

    with st.form("generation simulator"):
        generated_df = simulate_generation_mistakes(
            target_df=piece_split.target_df,
        )
        submitted = st.form_submit_button("Submit")

    if not submitted:
        st.write("Submit noise settings to continue")
        return

    generated_piece = ff.MidiPiece(
        df=generated_df,
    )

    cols = st.columns(3)
    with cols[0]:
        st.write("Input")
        streamlit_pianoroll.from_fortepyan(
            piece=source_piece,
            secondary_piece=target_piece,
        )

    with cols[1]:
        st.write("Output")
        streamlit_pianoroll.from_fortepyan(
            piece=source_piece,
            secondary_piece=generated_piece,
        )

    with cols[2]:
        st.write("Target vs Generation")
        streamlit_pianoroll.from_fortepyan(
            piece=target_piece,
            secondary_piece=generated_piece,
        )

    metrics_manager = MetricsManager.load_default()

    metric_results = metrics_manager.calculate_all(
        target_df=piece_split.target_df,
        generated_df=generated_df,
    )
    for metric_result in metric_results:
        n_metrics = len(metric_result.metrics)
        cols = st.columns(n_metrics + 1)
        cols[0].write(f"**{metric_result.name}:**")

        for it, (metric_name, metric) in enumerate(metric_result.metrics.items()):
            cols[it + 1].metric(
                label=metric_name,
                value=round(metric, 2),
            )


def simulate_generation_mistakes(target_df: pd.DataFrame) -> pd.DataFrame:
    generated_df = target_df.copy()
    cols = st.columns(3)

    # Modify the pitch value, so it's different than target
    add_pitch_noise = cols[0].checkbox(
        label="add noise to pitch",
        value=True,
    )
    pitch_noise_percentage = cols[0].number_input(
        label="% notes affected",
        value=50,
        min_value=0,
        max_value=100,
        step=10,
        key="pitch",
    )
    if add_pitch_noise and pitch_noise_percentage > 0:
        n_notes = generated_df.shape[0] * pitch_noise_percentage / 100
        n_notes = int(n_notes)
        idxs = generated_df.sample(n_notes).index
        # Pitch noise can only change pitches to higher values
        # should be easy to add a random direction change if this becomes an issue
        pitch_noise = np.random.randint(
            low=1,
            high=12,
            size=n_notes,
        )
        generated_df.loc[idxs, "pitch"] += pitch_noise

    add_velocity_noise = cols[1].checkbox(
        label="add noise to velocity",
        value=True,
    )
    velocity_noise_percentage = cols[1].number_input(
        label="% notes affected",
        value=50,
        min_value=0,
        max_value=100,
        step=10,
        key="velocity",
    )
    if add_velocity_noise:
        n_notes = generated_df.shape[0] * velocity_noise_percentage / 100
        n_notes = int(n_notes)
        idxs = generated_df.sample(n_notes).index
        velocity_noise = np.random.randint(
            low=-10,
            high=10,
            size=n_notes,
        )
        generated_df.loc[idxs, "velocity"] += velocity_noise

    add_start_noise = cols[2].checkbox(
        label="add noise to start",
        value=True,
    )
    start_noise_percentage = cols[2].number_input(
        label="% notes affected",
        value=50,
        min_value=0,
        max_value=100,
        step=10,
        key="start",
    )
    if add_start_noise:
        n_notes = generated_df.shape[0] * start_noise_percentage / 100
        n_notes = int(n_notes)
        idxs = generated_df.sample(n_notes).index
        start_noise = (
            np.random.random(
                size=n_notes,
            )
            - 0.5
        )
        # Just want [-0.15 - 0.15] range
        generated_df.loc[idxs, "start"] += start_noise * 0.3

    return generated_df


if __name__ == "__main__":
    main()
