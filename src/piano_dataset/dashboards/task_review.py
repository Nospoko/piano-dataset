import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import load_dataset

from piano_dataset.piano_tasks import PianoTaskManager


def main():
    st.write("# PIANO Prediction Tasks")
    task_manager = PianoTaskManager.load_default()

    available_tasks = task_manager.list_task_names()
    task_name = st.selectbox(
        label="Select PIANO task",
        options=available_tasks,
    )
    piano_task = task_manager.get_task(task_name=task_name)
    dataset = load_dataset("epr-labs/maestro-sustain-v2", split="test+validation")
    record_idx = st.number_input(
        label="Record ID",
        min_value=0,
        max_value=len(dataset) - 1,
        value=99,
    )

    record = dataset[record_idx]
    piece = MidiPiece.from_huggingface(record)
    piece.time_shift(-piece.df.start.min())

    cols = st.columns(2)
    piece_start = cols[0].number_input(
        label="piece start",
        value=0,
        min_value=0,
        max_value=piece.size - 128,
    )
    piece_finish = cols[1].number_input(
        label="piece finish",
        value=128,
        min_value=0,
        max_value=piece.size,
    )

    piece = piece[piece_start:piece_finish]

    st.json(piece.source)

    st.write(
        f"""
        Tokens
        ```
        {piano_task.prefix_tokens}
        ```
        """
    )

    piece_split = piano_task.prompt_target_split(piece.df)

    source_piece = MidiPiece(df=piece_split.source_df.copy())
    target_piece = MidiPiece(df=piece_split.target_df.copy())

    cols = st.columns(2)
    with cols[0]:
        st.write("## Prompt")
        streamlit_pianoroll.from_fortepyan(source_piece)

    with cols[1]:
        st.write("## Target")
        streamlit_pianoroll.from_fortepyan(target_piece)

    st.write("## Combined")
    streamlit_pianoroll.from_fortepyan(
        piece=source_piece,
        secondary_piece=target_piece,
    )


if __name__ == "__main__":
    main()
