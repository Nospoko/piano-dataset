import numpy as np
import pandas as pd


def calculate_duration_distribution(
    df: pd.DataFrame,
    n_bins: int,
) -> np.ndarray:
    """Calculate duration distribution.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing 'start' and 'end' columns with timestamps
    n_bins : int
        Number of quantization bins for duration values

    Returns
    -------
    distribution: np.ndarray
        Normalized distribution of note durations
    """
    if len(df) == 0:
        return np.array([])

    notes = df.copy()
    # Calculate duration of each note
    notes["duration"] = df.end - df.start
    notes = notes.dropna(subset=["duration"])

    # FIXME Looks like "5" is a magic number that has
    # to be synchronised in multiple spots
    # Create histogram of durations
    hist, bin_edges = np.histogram(
        notes["duration"],
        bins=np.linspace(start=0, stop=5, num=n_bins),
        range=(0, notes["duration"].quantile(0.99)),
    )

    distribution = hist.astype(float)
    # Normalize distribution
    total = np.sum(distribution)
    if total > 0:
        distribution = distribution / total

    return distribution


def calculate_duration_metrics(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    n_bins: int = 100,
) -> dict:
    """
    Calculate correlation coefficient between duration distributions of two MIDI sequences.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing target notes with columns: pitch, velocity, start, end.
    generated_df : pd.DataFrame
        DataFrame containing generated notes with columns: pitch, velocity, start, end.
    n_bins: int
        Number of bins to quantize duration values with.

    Returns
    -------
    correlation : float
        Correlation coefficient ranging from -1 to 1.
    metrics : dict
        Additional metrics including duration distributions.
    """
    target_dist = calculate_duration_distribution(target_df, n_bins)
    generated_dist = calculate_duration_distribution(generated_df, n_bins)

    correlation = np.corrcoef(target_dist, generated_dist)[0, 1]

    metrics = {
        "correlation": correlation,
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "target_durations": np.count_nonzero(target_dist),
        "generated_durations": np.count_nonzero(generated_dist),
        "duration_range": list(np.linspace(0, 5, n_bins)),  # Duration values in seconds
    }

    return metrics
