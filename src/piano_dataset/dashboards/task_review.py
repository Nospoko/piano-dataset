import streamlit as st
import streamlit_pianoroll
from fortepyan import MidiPiece
from datasets import load_dataset

from piano_dataset import PianoTasks


def main():
    st.write("# PIANO dataset task review")

    available_tasks = PianoTasks.list_tasks()
    task_name = st.selectbox(
        label="Select PIANO task",
        options=available_tasks,
    )
    piano_task = PianoTasks.get_task(task_name=task_name)

    dataset = load_dataset("epr-labs/maestro-sustain-v2", split="test+validation")
    record_idx = st.number_input(
        label="Record ID",
        min_value=0,
        max_value=len(dataset) - 1,
        value=43,
    )

    record = dataset[record_idx]
    piece = MidiPiece.from_huggingface(record)
    piece.time_shift(-piece.df.start.min())

    st.json(piece.source)

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
