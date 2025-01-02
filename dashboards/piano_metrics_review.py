import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

from piano_dataset import PianoTasks
from dashboards.utils import dataset_configuration
from piano_metrics.piano_metric import MetricsRunner


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
        streamlit_pianoroll.from_fortepyan(
            piece=source_piece,
            secondary_piece=target_piece,
        )

    with cols[1]:
        streamlit_pianoroll.from_fortepyan(
            piece=source_piece,
            secondary_piece=generated_piece,
        )

    with cols[2]:
        streamlit_pianoroll.from_fortepyan(
            piece=target_piece,
            secondary_piece=generated_piece,
        )

    metrics_manager = MetricsRunner.load_default()

    metric_results = metrics_manager.calculate_all(
        target_df=piece_split.target_df,
        generated_df=generated_df,
    )
    for metric_class, metric_result in metric_results.items():
        n_metrics = len(metric_result.metrics)
        cols = st.columns(n_metrics + 1)
        cols[0].write(f"**{metric_class}:**")
        for it, (metric_name, metric) in enumerate(metric_result.metrics.items()):
            cols[it + 1].metric(
                label=metric_name,
                value=round(metric, 2),
            )


def simulate_generation_mistakes(target_df: pd.DataFrame) -> pd.DataFrame:
    # TODO Make a better UX for explorint generation mistakes
    # Apply some noise to the fake generated piece
    generated_df = target_df.copy()
    cols = st.columns(3)

    add_pitch_noise = cols[0].checkbox(
        label="add noise to pitch",
        value=True,
    )
    if add_pitch_noise:
        pitch_noise = np.random.randint(
            low=-1,
            high=1,
            size=generated_df.shape[0],
        )
        generated_df.pitch += pitch_noise

    add_velocity_noise = cols[1].checkbox(
        label="add noise to velocity",
        value=True,
    )
    if add_velocity_noise:
        velocity_noise = np.random.randint(
            low=-10,
            high=10,
            size=generated_df.shape[0],
        )
        generated_df.velocity += velocity_noise

    add_start_noise = cols[2].checkbox(
        label="add noise to start",
        value=True,
    )
    if add_start_noise:
        start_noise = np.random.random(
            size=generated_df.shape[0],
        )
        # Just want [0 - 0.15] range
        generated_df.start += start_noise * 0.15

    return generated_df


if __name__ == "__main__":
    main()
