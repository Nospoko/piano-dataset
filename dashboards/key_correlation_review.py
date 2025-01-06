import json
from typing import List

import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go

from piano_dataset.dashboards.utils import dataset_configuration
from piano_metrics.key_distribution import calculate_key_metrics


def plot_key_distribution(
    distribution: np.ndarray,
    key_names: List[str],
    title: str = "",
) -> go.Figure:
    """Create a bar plot of key distribution."""
    fig = go.Figure()

    # Split into major and minor keys
    major_keys = key_names[:12]
    minor_keys = key_names[12:]
    major_dist = distribution[:12]
    minor_dist = distribution[12:]

    # Add major keys
    fig.add_trace(
        go.Bar(x=major_keys, y=major_dist, name="Major Keys", marker_color="rgb(55, 83, 109)"),
    )

    # Add minor keys
    fig.add_trace(
        go.Bar(x=minor_keys, y=minor_dist, name="Minor Keys", marker_color="rgb(26, 118, 255)"),
    )

    fig.update_layout(
        title=title,
        xaxis_tickangle=-45,
        yaxis_title="Probability",
        barmode="group",
        showlegend=True,
    )

    return fig


def plot_key_correlation_heatmap(
    target_dist: np.ndarray,
    generated_dist: np.ndarray,
    key_names: List[str],
) -> go.Figure:
    """Create a heatmap comparing two key distributions."""
    correlation_matrix = np.outer(target_dist, generated_dist)

    fig = go.Figure(
        data=go.Heatmap(
            z=correlation_matrix,
            x=key_names,
            y=key_names,
            colorscale="Viridis",
        ),
    )

    fig.update_layout(
        title="Key Distribution Correlation Heatmap",
        xaxis_tickangle=-45,
        yaxis_tickangle=0,
    )

    return fig


def main():
    st.title("Key Distribution Analysis Dashboard")

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
        segment_duration = st.slider(
            "Segment Duration (seconds)",
            0.05,
            0.5,
            0.125,
            0.025,
        )
        use_weighted = st.checkbox(
            "Weight pitches by duration and velocity",
            value=True,
            help="""
Controls how different notes influence key detection:
- When enabled: Weights key detection based on duration and velocity
- When disabled: All notes contribute equally to key detection

Example: With weighting enabled, a half note with velocity 100
will have more influence on determining the key than an eighth note with velocity 50.""",
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
        with st.spinner("Analyzing key distributions..."):
            key_metrics = calculate_key_metrics(
                target_df=piece1.df,
                generated_df=piece2.df,
                segment_duration=segment_duration,
                use_weighted=use_weighted,
            )

        # Display results
        st.header("Analysis Results")
        correlation = key_metrics["correlation"]
        st.metric("Correlation Coefficient", f"{correlation:.3f}")

        # Key distributions
        st.subheader("Key Distributions")
        col1, col2 = st.columns(2)

        with col1:
            st.write("Top keys in first piece:", ", ".join(key_metrics["target_top_keys"]))
            fig1 = plot_key_distribution(
                key_metrics["target_distribution"],
                list(key_metrics["key_names"].values()),
                "Key Distribution 1",
            )
            st.plotly_chart(fig1)

        with col2:
            st.write("Top keys in second piece:", ", ".join(key_metrics["generated_top_keys"]))
            fig2 = plot_key_distribution(
                key_metrics["generated_distribution"],
                list(key_metrics["key_names"].values()),
                "Key Distribution 2",
            )
            st.plotly_chart(fig2)

        # Correlation heatmap
        st.subheader("Key Distribution Correlation")
        fig3 = plot_key_correlation_heatmap(
            key_metrics["target_distribution"],
            key_metrics["generated_distribution"],
            list(
                key_metrics["key_names"].values(),
            ),
        )
        st.plotly_chart(fig3)

    else:
        st.error("Could not find selected pieces in dataset")


if __name__ == "__main__":
    main()
