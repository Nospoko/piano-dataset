import json

import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go

from dashboards.utils import dataset_configuration
from piano_metrics.duration_distribution import calculate_duration_metrics


def plot_duration_distribution(
    distribution: np.ndarray,
    n_bins: int,
    title: str = "",
) -> go.Figure:
    """Create a bar plot of duration distribution."""
    # Create bin edges for x-axis (durations from 0 to 5 seconds)
    bin_edges = np.linspace(start=0, stop=5, num=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=distribution,
            name="Duration Distribution",
            marker_color="rgb(109, 55, 109)",
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Note Duration (seconds)",
        yaxis_title="Probability",
        showlegend=False,
    )
    return fig


def plot_duration_correlation_heatmap(
    target_dist: np.ndarray,
    generated_dist: np.ndarray,
    n_bins: int,
) -> go.Figure:
    """Create a heatmap comparing two duration distributions."""
    correlation_matrix = np.outer(target_dist, generated_dist)
    bin_edges = np.linspace(start=0, stop=5, num=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=bin_centers,
            y=bin_centers,
            colorscale="Viridis",
        ),
    )

    fig.update_layout(
        title="Duration Distribution Correlation Heatmap",
        xaxis_title="Piece 2 Duration (seconds)",
        yaxis_title="Piece 1 Duration (seconds)",
    )
    return fig


def main():
    st.title("Note Duration Distribution Analysis Dashboard")

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
    n_bins = st.slider(
        "Number of duration bins",
        min_value=20,
        max_value=200,
        value=100,
        help="""
Controls the granularity of duration analysis:
- Higher values show more detailed duration patterns
- Lower values provide smoother, more general distribution

Example: 100 bins splits the 0-5 second range into 100 equal intervals.
Common note durations:
- Sixteenth note ≈ 0.125s at 120 BPM
- Eighth note ≈ 0.25s
- Quarter note ≈ 0.5s
- Half note ≈ 1.0s
- Whole note ≈ 2.0s""",
    )

    # Visualization of pieces
    st.header("Piece Visualizations")
    col1, col2 = st.columns(2)
    with col1:
        streamlit_pianoroll.from_fortepyan(piece=piece1)
    with col2:
        right_key = "right-" + json.dumps(piece2.source)
        streamlit_pianoroll.from_fortepyan(piece=piece2, key=right_key)

    # Analyze pieces
    with st.spinner("Analyzing duration distributions..."):
        duration_metrics = calculate_duration_metrics(
            target_df=piece1.df,
            generated_df=piece2.df,
            n_bins=n_bins,
        )

    # Display results
    st.header("Analysis Results")
    correlation = duration_metrics["correlation"]
    st.metric("Duration Correlation Coefficient", f"{correlation:.3f}")

    # Additional metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Active Duration Bins in Piece 1", duration_metrics["target_durations"])
    with col2:
        st.metric("Active Duration Bins in Piece 2", duration_metrics["generated_durations"])
    with col3:
        max_duration = max(
            piece1.df["end"].max() - piece1.df["start"].min(),
            piece2.df["end"].max() - piece2.df["start"].min(),
        )
        st.metric("Max Note Duration", f"{max_duration:.2f}s")

    # Duration distributions
    st.subheader("Duration Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_duration_distribution(
            distribution=duration_metrics["target_distribution"],
            n_bins=n_bins,
            title="Duration Distribution 1",
        )
        st.plotly_chart(fig1)
    with col2:
        fig2 = plot_duration_distribution(
            distribution=duration_metrics["generated_distribution"],
            n_bins=n_bins,
            title="Duration Distribution 2",
        )
        st.plotly_chart(fig2)

    # Correlation heatmap
    st.subheader("Duration Distribution Correlation")
    fig3 = plot_duration_correlation_heatmap(
        duration_metrics["target_distribution"],
        duration_metrics["generated_distribution"],
        n_bins,
    )
    st.plotly_chart(fig3)

    # Duration analysis insights
    st.subheader("Duration Analysis Insights")
    col1, col2 = st.columns(2)
    with col1:
        mean_duration1 = (piece1.df["end"] - piece1.df["start"]).mean()
        std_duration1 = (piece1.df["end"] - piece1.df["start"]).std()
        st.metric("Mean Duration (Piece 1)", f"{mean_duration1:.3f}s")
        st.metric("Std Duration (Piece 1)", f"{std_duration1:.3f}s")
    with col2:
        mean_duration2 = (piece2.df["end"] - piece2.df["start"]).mean()
        std_duration2 = (piece2.df["end"] - piece2.df["start"]).std()
        st.metric("Mean Duration (Piece 2)", f"{mean_duration2:.3f}s")
        st.metric("Std Duration (Piece 2)", f"{std_duration2:.3f}s")


if __name__ == "__main__":
    main()
