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


def test_modulation_to_dominant(spiral):
    notes = []
    notes.extend(create_chord([60, 64, 67], 0.0, 2.0))  # C major
    notes.extend(create_chord([67, 71, 74], 2.0, 2.0))  # G major
    notes.extend(create_chord([62, 66, 69], 4.0, 2.0))  # D major
    notes.extend(create_chord([67, 71, 74], 6.0, 2.0))  # G major
    df = pd.DataFrame(notes)
    analysis = analyze_piece(spiral, df, segment_duration=2.0)
    assert analysis["segment_keys"][0:2] == ["C major", "G major"]
    assert "G major" in analysis["top_keys"]


def test_weighted_velocity(spiral):
    notes = []
    # C major with high velocity
    notes.extend(create_chord([60, 64, 67], 0.0, 1.0, velocity=120))
    # G major with low velocity
    notes.extend(create_chord([67, 71, 74], 1.0, 1.0, velocity=30))
    df = pd.DataFrame(notes)
    key_name, _ = detect_key_from_notes(spiral, df)
    assert key_name == "C major"


def test_chromatic_passage(spiral):
    notes = []
    # Chromatic scale C to B
    for i in range(12):
        notes.append(create_note(60 + i, i * 0.5, 0.5))
    df = pd.DataFrame(notes)
    key_name, probs = detect_key_from_notes(spiral, df)
    max_prob = np.max(probs)
    assert max_prob < 0.4, "Chromatic passage should have ambiguous key"


def test_overlapping_notes(spiral):
    notes = []
    # C major chord with overlapping notes
    notes.append({"pitch": 60, "start": 0.0, "end": 2.0, "velocity": 64})  # C
    notes.append({"pitch": 64, "start": 0.5, "end": 2.5, "velocity": 64})  # E
    notes.append({"pitch": 67, "start": 1.0, "end": 3.0, "velocity": 64})  # G
    df = pd.DataFrame(notes)
    key_name, _ = detect_key_from_notes(spiral, df)
    assert key_name == "C major"


def test_parallel_keys(spiral):
    """Test detection of minor key with major dominant."""
    notes = []
    # C minor progression emphasizing dominant-tonic relationship
    notes.extend(create_chord([60, 63, 67], 0.0, 1.0))  # Cm
    notes.extend(create_chord([67, 71, 74], 1.0, 1.0))  # G (Stronger dominant)
    notes.extend(create_chord([65, 68, 72], 2.0, 1.0))  # Fm
    notes.extend(create_chord([60, 63, 67], 3.0, 1.0))  # Cm
    df = pd.DataFrame(notes)
    key_name, probs = detect_key_from_notes(spiral, df)
    assert key_name == "C minor"


def test_modal_mixture(spiral):
    """Test key detection with modal mixture, emphasizing major tonality."""
    notes = []
    # More emphasis on major tonic and dominant
    notes.extend(create_chord([60, 64, 67], 0.0, 2.0))  # C major (longer duration)
    notes.extend(create_chord([65, 68, 72], 2.0, 1.0))  # Fm
    notes.extend(create_chord([67, 71, 74], 3.0, 1.0))  # G major
    notes.extend(create_chord([60, 64, 67], 4.0, 2.0))  # C major (longer duration)

    df = pd.DataFrame(notes)
    key_name, _ = detect_key_from_notes(spiral, df)
    assert key_name == "C major"


def test_all_major_keys(spiral):
    """Test key detection for all major keys using stronger cadential patterns."""
    major_progressions = {
        "C": ([60, 64, 67], [67, 71, 74], [65, 69, 72]),  # C-G-F
        "G": ([67, 71, 74], [62, 66, 69], [60, 64, 67]),  # G-D-C
        "D": ([62, 66, 69], [69, 73, 76], [67, 71, 74]),  # D-A-G
        "A": ([69, 73, 76], [64, 68, 71], [62, 66, 69]),  # A-E-D
        "E": ([64, 68, 71], [71, 75, 78], [69, 73, 76]),  # E-B-A
        "B": ([71, 75, 78], [66, 70, 73], [64, 68, 71]),  # B-F#-E
        "F#": ([66, 70, 73], [73, 77, 80], [71, 75, 78]),  # F#-C#-B
        "C#": ([61, 65, 68], [68, 72, 75], [66, 70, 73]),  # C#-G#-F#
        "F": ([65, 69, 72], [60, 64, 67], [70, 74, 77]),  # F-C-A#
        "A#": ([70, 74, 77], [65, 69, 72], [63, 67, 70]),  # A#-F-D#
        "D#": ([63, 67, 70], [70, 74, 77], [68, 72, 75]),  # D#-A#-G#
        "G#": ([68, 72, 75], [63, 67, 70], [61, 65, 68]),  # G#-D#-C#
    }

    for root, (I, V, IV) in major_progressions.items():
        notes = []
        notes.extend(create_chord(I, 0.0, 2.0))  # I (longer)
        notes.extend(create_chord(V, 2.0, 1.0))  # V
        notes.extend(create_chord(IV, 3.0, 1.0))  # IV
        notes.extend(create_chord(I, 4.0, 2.0))  # I (longer)

        df = pd.DataFrame(notes)
        key_name, _ = detect_key_from_notes(spiral, df)
        assert key_name == f"{root} major", f"Failed to detect {root} major key"


def test_all_minor_keys(spiral):
    """Test key detection for all minor keys using their primary cadences."""
    minor_progressions = {
        "A": ([57, 60, 64], [62, 65, 69], [64, 68, 71]),  # Am-Dm-E
        "E": ([64, 67, 71], [69, 72, 76], [71, 75, 78]),  # Em-Am-B
        "B": ([71, 74, 78], [64, 67, 71], [66, 70, 73]),  # Bm-Em-F#
        "F#": ([66, 69, 73], [71, 74, 78], [73, 77, 80]),  # F#m-Bm-C#
        "C#": ([61, 64, 68], [66, 69, 73], [68, 72, 75]),  # C#m-F#m-G#
        "G#": ([68, 71, 75], [61, 64, 68], [63, 67, 70]),  # G#m-C#m-D#
        "D#": ([63, 66, 70], [68, 71, 75], [70, 74, 77]),  # D#m-G#m-A#
        "D": ([62, 65, 69], [67, 70, 74], [69, 73, 76]),  # Dm-Gm-A
        "G": ([67, 70, 74], [60, 63, 67], [62, 66, 69]),  # Gm-Cm-D
        "C": ([60, 63, 67], [65, 68, 72], [67, 71, 74]),  # Cm-Fm-G
        "F": ([65, 68, 72], [70, 73, 77], [72, 76, 79]),  # Fm-Bbm-C
        "A#": ([70, 73, 77], [63, 66, 70], [65, 69, 72]),  # Bbm-Ebm-F
    }

    for root, (i, iv, V) in minor_progressions.items():
        notes = []
        notes.extend(create_chord(i, 0.0, 1.0))  # i
        notes.extend(create_chord(iv, 1.0, 1.0))  # iv
        notes.extend(create_chord(V, 2.0, 1.0))  # V
        notes.extend(create_chord(i, 3.0, 1.0))  # i

        df = pd.DataFrame(notes)
        key_name, _ = detect_key_from_notes(spiral, df)
        assert key_name == f"{root} minor", f"Failed to detect {root} minor key"


def test_problematic_keys(spiral):
    """Test specifically challenging keys for the Spiral Array model."""
    problem_cases = {
        # Simpler progressions for trouble spots
        "F#": ([66, 70, 73], [73, 77, 80]),  # F# -> C# (fifth)
        "F": ([65, 69, 72], [72, 76, 79]),  # F -> C (fifth)
        "D#": ([63, 67, 70], [70, 74, 77]),  # D# -> A# (fifth)
    }

    for root, (I, V) in problem_cases.items():
        notes = []
        # Use only tonic-dominant relationship with strong emphasis on tonic
        notes.extend(create_chord(I, 0.0, 3.0))  # Tonic (longer)
        notes.extend(create_chord(V, 3.0, 1.0))  # Dominant (shorter)
        notes.extend(create_chord(I, 4.0, 2.0))  # Tonic (again)

        df = pd.DataFrame(notes)
        key_name, probs = detect_key_from_notes(spiral, df)
        print(f"\nFor {root} major:")  # Let's see the probability distribution
        top_3 = sorted([(spiral.key_names[i], probs[i]) for i in range(24)], key=lambda x: x[1], reverse=True)[:3]
        print(f"Top 3 candidates: {top_3}")
        assert key_name == f"{root} major", f"Failed to detect {root} major key"
