import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll

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
    target_piece = piece[piece_start:piece_finish]

    # This makes a copy of the DataFrame inside the piece
    generated_piece = target_piece[:]

    # Apply some noise to the fake generated piece
    pitch_noise = np.random.randint(
        low=-1,
        high=1,
        size=generated_piece.size,
    )
    generated_piece.df.pitch += pitch_noise

    cols = st.columns(2)
    with cols[0]:
        streamlit_pianoroll.from_fortepyan(target_piece)

    with cols[1]:
        streamlit_pianoroll.from_fortepyan(generated_piece)

    metrics_manager = MetricsRunner.load_default()

    metric_results = metrics_manager.calculate_all(
        target_df=target_piece.df,
        generated_df=generated_piece.df,
    )
    for metric_name, metric_result in metric_results.items():
        st.metric(
            label=metric_name,
            value=round(metric_result.value, 2),
        )


if __name__ == "__main__":
    main()
