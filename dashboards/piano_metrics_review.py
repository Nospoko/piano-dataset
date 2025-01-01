import numpy as np
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

    # This makes a copy of the DataFrame inside the piece
    generated_df = piece_split.target_df.copy()

    # TODO Make a better UX for explorint generation mistakes
    # Apply some noise to the fake generated piece
    pitch_noise = np.random.randint(
        low=-1,
        high=1,
        size=generated_df.shape[0],
    )
    generated_df.pitch += pitch_noise
    velocity_noise = np.random.randint(
        low=-10,
        high=10,
        size=generated_df.shape[0],
    )
    generated_df.velocity += velocity_noise
    start_noise = np.random.random(
        size=generated_df.shape[0],
    )
    # Just want [0 - 0.15] range
    generated_df.start += start_noise * 0.15

    generated_piece = ff.MidiPiece(
        df=generated_df,
    )

    cols = st.columns(2)
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


if __name__ == "__main__":
    main()
