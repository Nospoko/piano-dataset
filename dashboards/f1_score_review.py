import numpy as np
import pandas as pd
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go
from utils import dataset_configuration

from piano_metrics.f1_piano import calculate_f1


def plot_f1_time_series(metrics: dict, title: str = "") -> go.Figure:
    """Create line plot of F1 scores over time."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=metrics["time_points"], y=metrics["f1"], name="F1", line=dict(color="rgb(55, 83, 109)")),
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
        go.Scatter(x=metrics["time_points"], y=metrics["recall"], name="Recall", line=dict(color="rgb(98, 182, 149)")),
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

    col1, col2 = st.columns(2)
    with col1:
        dataset_1 = dataset_configuration(key="0")
    with col2:
        dataset_2 = dataset_configuration(key="1")

    if len(dataset_1) > 0 and len(dataset_2) > 0:
        piece1 = ff.MidiPiece.from_huggingface(dataset_1[0])
        piece2 = ff.MidiPiece.from_huggingface(dataset_2[0])

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
            "Use pitch class matching",
            value=True,
        )

        # Visualization of pieces
        st.header("Piece Visualizations")
        col1, col2 = st.columns(2)
        with col1:
            streamlit_pianoroll.from_fortepyan(piece=piece1)
        with col2:
            streamlit_pianoroll.from_fortepyan(piece=piece2, secondary_piece=piece1)

        # Analyze pieces
        with st.spinner("Calculating F1 scores..."):
            f1_score, metrics = calculate_f1(
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
            st.metric("F1 Score", f"{f1_score:.3f}")
        with col2:
            st.metric("Average Precision", f"{np.mean(metrics['precision']):.3f}")
        with col3:
            st.metric("Average Recall", f"{np.mean(metrics['recall']):.3f}")

        # Time series visualization
        st.subheader("Score Evolution Over Time")
        fig = plot_f1_time_series(metrics, "F1, Precision, and Recall Over Time")
        st.plotly_chart(fig)

        # Detailed analysis
        st.subheader("Detailed Analysis")
        df = pd.DataFrame(
            {
                "Time": metrics["time_points"],
                "Duration (ms)": [d * min_time_unit * 1000 for d in metrics["durations"]],
                "F1": metrics["f1"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
            }
        ).round(3)
        st.dataframe(df)

    else:
        st.error("Could not find selected pieces in dataset")


if __name__ == "__main__":
    main()
