import json

import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go

from piano_metrics.f1_piano import calculate_f1
from piano_dataset.dashboards.utils import dataset_configuration


def plot_f1_time_series(metrics: dict, title: str = "") -> go.Figure:
    """Create line plot of F1 scores over time."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=metrics["time_points"],
            y=metrics["f1"],
            name="F1",
            line=dict(color="rgb(55, 83, 109)"),
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["time_points"],
            y=metrics["precision"],
            name="Precision",
            line=dict(color="rgb(26, 118, 255)"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=metrics["time_points"],
            y=metrics["recall"],
            name="Recall",
            line=dict(color="rgb(98, 182, 149)"),
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time (s)",
        yaxis_title="Score",
        showlegend=True,
    )
    return fig


def main():
    st.title("F1 Score Analysis Dashboard")

    st.markdown("### Piece Selectors")
    col1, col2 = st.columns(2)
    with col1:
        dataset_1 = dataset_configuration(key="0")
        record_id_1 = st.number_input(
            label=f"Record number [0-{len(dataset_1)}]",
            value=0,
            key="record_id_0",
        )
    with col2:
        dataset_2 = dataset_configuration(key="1")
        record_id_2 = st.number_input(
            label=f"Record number [0-{len(dataset_2)}]",
            value=0,
            key="record_id_1",
        )

    if len(dataset_1) == 0 or len(dataset_2) == 0:
        st.error("Could not find selected pieces in dataset")
        return

    piece1 = ff.MidiPiece.from_huggingface(dataset_1[record_id_1])
    piece2 = ff.MidiPiece.from_huggingface(dataset_2[record_id_2])
    # Analysis parameters
    st.header("Analysis Parameters")
    min_time_unit = st.number_input(
        "Minimum Time Unit (s)",
        value=0.01,
        step=0.001,
        format="%.3f",
    )
    velocity_threshold = st.number_input(
        "Velocity Threshold",
        value=30,
        step=1,
    )
    use_pitch_class = st.checkbox(
        "Match pitch classes (ignore octaves)",
        value=True,
        help="""
Controls how pitches are matched between pieces:
- When enabled: Notes are considered matching if they have the same pitch class
(e.g., all C notes match regardless of octave)
- When disabled: Notes must have exactly the same MIDI pitch to match (e.g., C4 only matches C4)

Example: With pitch class matching, C4 matches C3, C5 etc. Without it, C4 only matches C4.""",
    )

    # Visualization of pieces
    st.header("Piece Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        streamlit_pianoroll.from_fortepyan(piece=piece1)
    with col2:
        right_key = "right-" + json.dumps(piece2.source) + json.dumps(piece1.source)
        streamlit_pianoroll.from_fortepyan(
            piece=piece2,
            secondary_piece=piece1,
            key=right_key,
        )

    # Analyze pieces
    with st.spinner("Calculating F1 scores..."):
        f1_metrics = calculate_f1(
            target_df=piece1.df,
            generated_df=piece2.df,
            min_time_unit=min_time_unit,
            velocity_threshold=velocity_threshold,
            use_pitch_class=use_pitch_class,
        )

    # Display results
    st.header("Analysis Results")
    col1, col2, col3 = st.columns(3)
    with col1:
        weighted_f1 = f1_metrics["weighted_f1"]
        st.metric("Weighted F1 Score", f"{weighted_f1:.3f}")
    with col2:
        st.metric("Average Precision", f"{np.mean(f1_metrics['precision']):.3f}")
    with col3:
        st.metric("Average Recall", f"{np.mean(f1_metrics['recall']):.3f}")

    # Time series visualization
    st.subheader("Score Evolution Over Time")
    fig = plot_f1_time_series(f1_metrics, "F1, Precision, and Recall Over Time")
    st.plotly_chart(fig)

    # Detailed analysis
    st.subheader("Detailed Analysis")
    df = pd.DataFrame(
        {
            "Time": f1_metrics["time_points"],
            "Duration (ms)": [d * min_time_unit * 1000 for d in f1_metrics["durations"]],
            "F1": f1_metrics["f1"],
            "Precision": f1_metrics["precision"],
            "Recall": f1_metrics["recall"],
        }
    ).round(3)
    st.dataframe(df)


if __name__ == "__main__":
    main()
