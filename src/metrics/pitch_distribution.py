from typing import Dict, Tuple

import numpy as np
import pandas as pd


def calculate_pitch_weights(df: pd.DataFrame, use_weighted: bool = True) -> np.ndarray:
    """Calculate weighted pitch distribution across 88 piano keys (21-108 MIDI range)"""
    distribution = np.zeros(88)

    if len(df) == 0:
        return distribution

    for _, note in df.iterrows():
        pitch_idx = int(note["pitch"]) - 21  # Convert MIDI pitch to piano key index
        if 0 <= pitch_idx < 88:  # Only count pitches in piano range
            weight = 1.0
            if use_weighted:
                duration = note["end"] - note["start"]
                velocity = note["velocity"] / 127.0
                weight = duration * velocity
            distribution[pitch_idx] += weight

    # Normalize distribution
    total = np.sum(distribution)
    if total > 0:
        distribution = distribution / total

    return distribution


def calculate_pitch_correlation(
    target_df: pd.DataFrame, generated_df: pd.DataFrame, use_weighted: bool = True
) -> Tuple[float, Dict]:
    """
    Calculate correlation coefficient between pitch distributions of two MIDI sequences.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing target notes with columns: pitch, velocity, start, end.
    generated_df : pd.DataFrame
        DataFrame containing generated notes with columns: pitch, velocity, start, end.
    use_weighted : bool
        If True, weight pitch contributions by note duration and velocity.

    Returns
    -------
    correlation : float
        Correlation coefficient ranging from -1 to 1.
    metrics : dict
        Additional metrics including pitch distributions.
    """
    target_dist = calculate_pitch_weights(target_df, use_weighted)
    generated_dist = calculate_pitch_weights(generated_df, use_weighted)

    correlation = np.corrcoef(target_dist, generated_dist)[0, 1]

    metrics = {
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "target_active_pitches": np.count_nonzero(target_dist),
        "generated_active_pitches": np.count_nonzero(generated_dist),
        "pitch_range": list(range(21, 109)),  # MIDI pitch numbers
    }

    return correlation, metrics
