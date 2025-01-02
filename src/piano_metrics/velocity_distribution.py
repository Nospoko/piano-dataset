import numpy as np
import pandas as pd

from piano_metrics.distribution_metrics import calculate_distribution_metrics


def calculate_velocity_weights(
    notes_df: pd.DataFrame,
    use_weighted: bool = True,
) -> np.ndarray:
    """Calculate weighted velocity distribution across 128 possible velocity values."""
    distribution = np.zeros(128)

    if len(notes_df) == 0:
        return distribution

    for _, note in notes_df.iterrows():
        weight = 1.0
        velocity = int(note["velocity"])
        if use_weighted:
            duration = note["end"] - note["start"]
            weight = duration
        distribution[velocity] += weight

    # Normalize distribution
    total = np.sum(distribution)
    if total > 0:
        distribution = distribution / total

    return distribution


def calculate_velocity_metrics(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    use_weighted: bool = True,
) -> dict:
    """
    Calculate comparison metrics between velocity distributions of two MIDI sequences.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing target notes with columns: pitch, velocity, start, end.
    generated_df : pd.DataFrame
        DataFrame containing generated notes with columns: pitch, velocity, start, end.
    use_weighted : bool
        If True, weight velocity contributions by note duration.

    Returns
    -------
    correlation : float
        Correlation coefficient ranging from -1 to 1.
    metrics : dict
        Additional metrics including velocity distributions.
    """
    target_dist = calculate_velocity_weights(
        notes_df=target_df,
        use_weighted=use_weighted,
    )
    generated_dist = calculate_velocity_weights(
        notes_df=generated_df,
        use_weighted=use_weighted,
    )

    distribution_metrics = calculate_distribution_metrics(
        target_dist=target_dist,
        generated_dist=generated_dist,
    )

    metadata = {
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "target_velocities": np.count_nonzero(target_dist),
        "generated_velocities": np.count_nonzero(generated_dist),
        "velocity_range": list(range(0, 127)),  # MIDI velocity numbers
    }

    result = {
        "distribution_metrics": distribution_metrics,
        "metadata": metadata,
    }

    return result
