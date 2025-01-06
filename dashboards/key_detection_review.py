import numpy as np
import fortepyan as ff
import streamlit as st
import streamlit_pianoroll
import plotly.graph_objects as go
from datasets import load_dataset

from piano_dataset.dashboards.utils import select_part_dataset
from piano_metrics.key_distribution import SpiralArray, detect_key_from_notes


def analyze_key_timeline(piece: ff.MidiPiece, segment_duration: float = 0.125) -> dict:
    """Analyze key probabilities over time for a piece."""
    spiral = SpiralArray()
    total_duration = piece.df["end"].max()

    # Initialize timeline data
    timeline_data = {"times": [], "key_probs": [], "top_keys": []}

    current_time = 0
    while current_time < total_duration:
        # Get key probabilities for this segment
        key_name, key_probs = detect_key_from_notes(
            spiral=spiral,
            notes_df=piece.df,
            segment_start=current_time,
            segment_duration=segment_duration,
            use_weighted=True,
        )

        # Store results
        timeline_data["times"].append(current_time)
        timeline_data["key_probs"].append(key_probs)
        timeline_data["top_keys"].append(key_name)

        current_time += segment_duration

    return timeline_data


def plot_key_timeline(timeline_data: dict, key_names: dict, num_top_keys: int = 5) -> go.Figure:
    """Create an interactive plot showing key probability changes over time."""
    # Convert probabilities to numpy array for easier manipulation
    probs_array = np.array(timeline_data["key_probs"])

    # Get indices of top keys across all time segments
    mean_probs = np.mean(probs_array, axis=0)
    top_key_indices = np.argsort(mean_probs)[-num_top_keys:]

    # Create figure
    fig = go.Figure()

    # Add a line for each top key
    for idx in top_key_indices:
        key_name = key_names[idx]
        fig.add_trace(
            go.Scatter(
                x=timeline_data["times"],
                y=probs_array[:, idx],
                name=key_name,
                mode="lines",
                hovertemplate=f"{key_name}<extra></extra>",
            )
        )

    fig.update_layout(
        title="Key Probability Timeline",
        xaxis_title="Time (seconds)",
        yaxis_title="Probability",
        yaxis_range=[0, 1],
        hovermode="x unified",
        showlegend=True,
    )

    return fig


def main():
    st.title("Key Timeline Analysis Dashboard")

    st.header("Dataset Configuration")
    dataset_path = st.text_input(
        "Dataset Path",
        value="epr-labs/maestro-sustain-v2",
        help="Enter the path to the dataset",
    )
    dataset_split = st.selectbox(
        "Dataset Split",
        options=["validation", "train", "test"],
        help="Choose the dataset split to use",
    )

    dataset = load_dataset(
        dataset_path,
        split=dataset_split,
        trust_remote_code=True,
    )

    dataset = select_part_dataset(midi_dataset=dataset, key="0")
    record_id = st.number_input(
        label=f"Record number [0-{len(dataset)}]",
        value=0,
    )

    if len(dataset) > 0:
        piece = ff.MidiPiece.from_huggingface(dataset[record_id])

        # Analysis parameters
        st.header("Analysis Parameters")
        segment_duration = st.number_input(
            label="Segment Duration (seconds)",
            value=0.05,
            help="Time window for analyzing key probabilities",
        )
        num_top_keys = st.number_input(
            "Number of top keys to display",
            value=5,
            max_value=24,
            help="Number of most probable keys to show in the timeline",
        )

        st.header("Piano Roll Visualization")
        streamlit_pianoroll.from_fortepyan(piece=piece)

        # Analyze and visualize key timeline
        with st.spinner("Analyzing key probabilities..."):
            spiral = SpiralArray()
            timeline_data = analyze_key_timeline(
                piece=piece,
                segment_duration=segment_duration,
            )
            fig = plot_key_timeline(
                timeline_data=timeline_data,
                key_names=spiral.key_names,
                num_top_keys=num_top_keys,
            )

            # Display results
            st.header("Key Timeline Analysis")
            st.plotly_chart(fig, use_container_width=True)

            # Display statistics
            st.subheader("Analysis Statistics")
            col1, col2 = st.columns(2)

            with col1:
                total_segments = len(timeline_data["times"])
                st.metric("Total Time Segments", total_segments)

            with col2:
                most_common_key = max(set(timeline_data["top_keys"]), key=timeline_data["top_keys"].count)
                st.metric("Most Common Key", most_common_key)
    else:
        st.error("Could not find selected piece in dataset")


if __name__ == "__main__":
    main()
