import json

import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go

from dashboards.utils import dataset_configuration
from piano_metrics.velocity_distribution import calculate_velocity_metrics


def plot_velocity_distribution(
    distribution: np.ndarray,
    title: str = "",
) -> go.Figure:
    """Create a bar plot of velocity distribution."""
    velocity_values = list(range(128))
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=velocity_values,
            y=distribution,
            name="Velocity Distribution",
            marker_color="rgb(109, 55, 83)",
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="MIDI Velocity",
        yaxis_title="Probability",
        showlegend=False,
    )
    return fig


def plot_velocity_correlation_heatmap(
    target_dist: np.ndarray,
    generated_dist: np.ndarray,
) -> go.Figure:
    """Create a heatmap comparing two velocity distributions."""
    correlation_matrix = np.outer(target_dist, generated_dist)
    velocity_values = list(range(128))

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=velocity_values,
            y=velocity_values,
            colorscale="Viridis",
        ),
    )

    fig.update_layout(
        title="Velocity Distribution Correlation Heatmap",
        xaxis_title="Piece 2 MIDI Velocity",
        yaxis_title="Piece 1 MIDI Velocity",
    )
    return fig


def main():
    st.title("Velocity Distribution Analysis Dashboard")

    st.markdown("### Piece Selectors")
    col1, col2 = st.columns(2)
    with col1:
        dataset_1 = dataset_configuration(key="0")
        record_id_1 = st.number_input(
            label=f"Record number dataset 1 [0-{len(dataset_1) - 1}]",
            value=0,
            min_value=0,
            max_value=len(dataset_1) - 1,
        )
    with col2:
        dataset_2 = dataset_configuration(key="1")
        record_id_2 = st.number_input(
            label=f"Record number dataset 2 [0-{len(dataset_2) - 1}]",
            value=0,
            min_value=0,
            max_value=len(dataset_2) - 1,
        )

    if len(dataset_1) == 0 or len(dataset_2) == 0:
        st.error("Could not find selected pieces in dataset")
        return

    piece1 = ff.MidiPiece.from_huggingface(dataset_1[record_id_1])
    piece2 = ff.MidiPiece.from_huggingface(dataset_2[record_id_2])

    # Analysis parameters
    st.header("Analysis Parameters")
    use_weighted = st.checkbox(
        "Weight velocity by duration",
        value=True,
        help="""
Controls how each note affects the velocity distribution:
- When enabled: Longer notes have stronger influence on the velocity distribution
- When disabled: All notes contribute equally regardless of length

Example: A half note at velocity 80 will contribute more
than an eighth note at velocity 80 when weighting is enabled.""",
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
    with st.spinner("Analyzing velocity distributions..."):
        velocity_metrics = calculate_velocity_metrics(
            target_df=piece1.df,
            generated_df=piece2.df,
            use_weighted=use_weighted,
        )

    # Display results
    st.header("Analysis Results")

    correlation = velocity_metrics["correlation"]
    taxicab_distance = velocity_metrics["taxicab_distance"]

    st.metric("Velocity Correlation Coefficient", f"{correlation:.3f}")
    st.metric("Velocity Taxicab", f"{taxicab_distance:.3f}")

    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Velocities in Piece 1", velocity_metrics["target_velocities"])
    with col2:
        st.metric("Active Velocities in Piece 2", velocity_metrics["generated_velocities"])

    # Velocity distributions
    st.subheader("Velocity Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_velocity_distribution(
            distribution=velocity_metrics["target_distribution"],
            title="Velocity Distribution 1",
        )
        st.plotly_chart(fig1)
    with col2:
        fig2 = plot_velocity_distribution(
            distribution=velocity_metrics["generated_distribution"],
            title="Velocity Distribution 2",
        )
        st.plotly_chart(fig2)

    # Correlation heatmap
    st.subheader("Velocity Distribution Correlation")
    fig3 = plot_velocity_correlation_heatmap(
        target_dist=velocity_metrics["target_distribution"],
        generated_dist=velocity_metrics["generated_distribution"],
    )
    st.plotly_chart(fig3)


if __name__ == "__main__":
    main()
