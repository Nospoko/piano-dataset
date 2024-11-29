from typing import List

import pytest
import numpy as np
import pandas as pd

from src.piano_metrics.key_distribution import SpiralArray, analyze_piece, detect_key_from_notes


def create_note(pitch: int, start: float, duration: float, velocity: int = 64) -> dict:
    return {"pitch": pitch, "start": start, "end": start + duration, "velocity": velocity}


def create_chord(pitches: List[int], start: float, duration: float, velocity: int = 64) -> List[dict]:
    return [create_note(pitch, start, duration, velocity) for pitch in pitches]


@pytest.fixture(scope="module")
def spiral():
    return SpiralArray()


def test_c_major(spiral):
    notes = []
    notes.extend(create_chord([60, 64, 67], 0.0, 1.0))  # C major
    notes.extend(create_chord([65, 69, 72], 1.0, 1.0))  # F major
    notes.extend(create_chord([67, 71, 74], 2.0, 1.0))  # G major
    notes.extend(create_chord([60, 64, 67], 3.0, 1.0))  # C major
    df = pd.DataFrame(notes)
    key_name, _ = detect_key_from_notes(spiral, df)
    assert key_name == "C major"


def test_a_minor(spiral):
    notes = []
    notes.extend(create_chord([57, 60, 64], 0.0, 1.0))  # Am
    notes.extend(create_chord([62, 65, 69], 1.0, 1.0))  # Dm
    notes.extend(create_chord([64, 68, 71], 2.0, 1.0))  # E
    notes.extend(create_chord([57, 60, 64], 3.0, 1.0))  # Am
    df = pd.DataFrame(notes)
    key_name, _ = detect_key_from_notes(spiral, df)
    assert key_name == "A minor"


def test_empty_input(spiral):
    df = pd.DataFrame(columns=["pitch", "start", "end", "velocity"])
    key_name, probs = detect_key_from_notes(spiral, df)
    assert key_name == "No key detected"
    assert np.allclose(probs, np.zeros(24))


def test_segment_analysis(spiral):
    notes = []
    notes.extend(create_chord([67, 71, 74], 0.0, 2.0))  # G major section
    notes.extend(create_chord([60, 64, 67], 2.0, 2.0))  # C major section

    df = pd.DataFrame(notes)

    analysis = analyze_piece(spiral, df, segment_duration=2.0)
    assert len(analysis["segment_keys"]) == 2
    assert analysis["segment_keys"][0] == "G major", analysis["segment_keys"]
    assert analysis["segment_keys"][1] == "C major", analysis["segment_keys"]


def test_pitch_spiral_mapping(spiral):
    pitches = [60, 64, 67]  # C, E, G
    durations = [1.0] * 3
    velocities = [1.0] * 3
    center = spiral.get_center_of_effect(pitches, durations, velocities)

    key_index, probs = spiral.get_key(center, return_distribution=True)
    sorted_keys = [(spiral.key_names[i], probs[i]) for i in range(24)]
    sorted_keys.sort(key=lambda x: x[1], reverse=True)

    assert sorted_keys[0][0] == "C major", f"Expected C major, but top 3 keys were: {sorted_keys[:3]}"


def test_c_major_debug(spiral):
    notes = []
    notes.extend(create_chord([60, 64, 67], 0.0, 1.0))  # C major
    df = pd.DataFrame(notes)
    key_name, probs = detect_key_from_notes(spiral, df)

    top_3 = [(spiral.key_names[i], probs[i]) for i in range(24)]
    top_3.sort(key=lambda x: x[1], reverse=True)
    assert key_name == "C major", f"Top 3 detected keys were: {top_3[:3]}"
