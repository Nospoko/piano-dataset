import json

import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go

from dashboards.utils import dataset_configuration
from piano_metrics.dstart_distribution import calculate_dstart_correlation


def plot_dstart_distribution(
    distribution: np.ndarray,
    n_bins: int,
    title: str = "",
) -> go.Figure:
    """Create a bar plot of dstart distribution."""
    # Create bin edges for x-axis (time differences from 0 to 2 seconds)
    bin_edges = np.linspace(start=0, stop=2, num=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=distribution,
            name="Start Time Distribution",
            marker_color="rgb(83, 109, 55)",
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time Difference Between Consecutive Notes (seconds)",
        yaxis_title="Probability",
        showlegend=False,
    )
    return fig


def plot_dstart_correlation_heatmap(
    target_dist: np.ndarray,
    generated_dist: np.ndarray,
    n_bins: int,
) -> go.Figure:
    """Create a heatmap comparing two dstart distributions."""
    correlation_matrix = np.outer(target_dist, generated_dist)
    bin_edges = np.linspace(start=0, stop=2, num=n_bins)
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
        title="Start Time Distribution Correlation Heatmap",
        xaxis_title="Piece 2 Time Differences (seconds)",
        yaxis_title="Piece 1 Time Differences (seconds)",
    )
    return fig


def main():
    st.title("Note Start Time Distribution Analysis Dashboard")

    st.markdown("### Piece Selectors")
    col1, col2 = st.columns(2)
    with col1:
        dataset_1 = dataset_configuration(key="0")
        record_id_1 = st.number_input(label=f"Record number [0-{len(dataset_1)}]", value=0, key="record_id_0")
    with col2:
        dataset_2 = dataset_configuration(key="1")
        record_id_2 = st.number_input(label=f"Record number [0-{len(dataset_2)}]", value=0, key="record_id_1")

    if len(dataset_1) > 0 and len(dataset_2) > 0:
        piece1 = ff.MidiPiece.from_huggingface(dataset_1[record_id_1])
        piece2 = ff.MidiPiece.from_huggingface(dataset_2[record_id_2])

        # Analysis parameters
        st.header("Analysis Parameters")
        n_bins = st.slider(
            "Number of time bins",
            min_value=10,
            max_value=100,
            value=50,
            help="""
Controls the granularity of time difference analysis:
- Higher values give more detailed timing distribution
- Lower values provide smoother, more general patterns

Example: 50 bins splits the 0-2 second range into 50 equal intervals.""",
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
        with st.spinner("Analyzing start time distributions..."):
            correlation, metrics = calculate_dstart_correlation(
                target_df=piece1.df,
                generated_df=piece2.df,
                n_bins=n_bins,
            )

        # Display results
        st.header("Analysis Results")
        st.metric("Start Time Correlation Coefficient", f"{correlation:.3f}")

        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Active Time Bins in Piece 1", metrics["target_dstarts"])
        with col2:
            st.metric("Active Time Bins in Piece 2", metrics["generated_dstarts"])

        # Start time distributions
        st.subheader("Dstart Time Distributions")
        col1, col2 = st.columns(2)
        with col1:
            fig1 = plot_dstart_distribution(metrics["target_distribution"], n_bins, "Start Time Distribution 1")
            st.plotly_chart(fig1)
        with col2:
            fig2 = plot_dstart_distribution(metrics["generated_distribution"], n_bins, "Start Time Distribution 2")
            st.plotly_chart(fig2)

        # Correlation heatmap
        st.subheader("Start Time Distribution Correlation")
        fig3 = plot_dstart_correlation_heatmap(
            metrics["target_distribution"],
            metrics["generated_distribution"],
            n_bins,
        )
        st.plotly_chart(fig3)

    else:
        st.error("Could not find selected pieces in dataset")


if __name__ == "__main__":
    main()
