import json

import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go

from piano_dataset.dashboards.utils import dataset_configuration
from piano_metrics.pitch_distribution import calculate_pitch_metrics


def plot_pitch_distribution(
    distribution: np.ndarray,
    title: str = "",
) -> go.Figure:
    """Create a bar plot of pitch distribution."""
    pitch_numbers = list(range(21, 109))
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=pitch_numbers,
            y=distribution,
            name="Pitch Distribution",
            marker_color="rgb(55, 83, 109)",
        ),
    )

    fig.update_layout(
        title=title,
        xaxis_title="MIDI Pitch Number",
        yaxis_title="Probability",
        showlegend=False,
    )
    return fig


def plot_pitch_correlation_heatmap(
    target_dist: np.ndarray,
    generated_dist: np.ndarray,
) -> go.Figure:
    """Create a heatmap comparing two pitch distributions."""
    correlation_matrix = np.outer(target_dist, generated_dist)
    pitch_numbers = list(range(21, 109))

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=pitch_numbers,
            y=pitch_numbers,
            colorscale="Viridis",
        ),
    )

    fig.update_layout(
        title="Pitch Distribution Correlation Heatmap",
        xaxis_title="Piece 2 MIDI Pitch",
        yaxis_title="Piece 1 MIDI Pitch",
    )
    return fig


def main():
    st.title("Pitch Distribution Analysis Dashboard")

    st.markdown("### Piece Selectors")
    col1, col2 = st.columns(2)
    with col1:
        dataset_1 = dataset_configuration(key="0")
        # TODO Use min/max_value arguments
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
    use_weighted = st.checkbox(
        "Weight pitch by duration and velocity",
        value=True,
        help="""
Controls how each note affects the pitch distribution:
- When enabled: Longer notes and higher velocity notes have stronger influence
- When disabled: All notes contribute equally regardless of length or velocity

Example: A half note at velocity 100 will contribute more
than an eighth note at velocity 60 when weighting is enabled.""",
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
    with st.spinner("Analyzing pitch distributions..."):
        pitch_metrics = calculate_pitch_metrics(
            target_df=piece1.df,
            generated_df=piece2.df,
            use_weighted=use_weighted,
        )

    # Display results
    st.header("Analysis Results")
    correlation = pitch_metrics["correlation"]
    st.metric("Correlation Coefficient", f"{correlation:.3f}")

    # Additional metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Active Pitches in Piece 1", pitch_metrics["target_active_pitches"])
    with col2:
        st.metric("Active Pitches in Piece 2", pitch_metrics["generated_active_pitches"])

    # Pitch distributions
    st.subheader("Pitch Distributions")
    col1, col2 = st.columns(2)
    with col1:
        fig1 = plot_pitch_distribution(
            distribution=pitch_metrics["target_distribution"],
            title="Pitch Distribution 1",
        )
        st.plotly_chart(fig1)
    with col2:
        fig2 = plot_pitch_distribution(
            distribution=pitch_metrics["generated_distribution"],
            title="Pitch Distribution 2",
        )
        st.plotly_chart(fig2)

    # Correlation heatmap
    st.subheader("Pitch Distribution Correlation")
    fig3 = plot_pitch_correlation_heatmap(
        target_dist=pitch_metrics["target_distribution"],
        generated_dist=pitch_metrics["generated_distribution"],
    )

    st.plotly_chart(fig3)


if __name__ == "__main__":
    main()
