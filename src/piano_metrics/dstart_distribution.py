import numpy as np
import pandas as pd

from piano_metrics.distribution_metrics import calculate_distribution_metrics


def calculate_dstart_distribution(
    df: pd.DataFrame,
    n_bins: int,
) -> np.ndarray:
    """Calculate dstart distribution.

    Parameters
    ----------
        df : pd.DataFrame
            DataFrame containing 'start' column with timestamps
        n_bins : int
            Number of quantization bins for dstart values

    Returns
    -------
        distribution: np.ndarray:
            Normalized distribution of time differences between consecutive starts
    """
    if len(df) == 0:
        return np.array([])

    notes = df.copy()
    # Calculate time differences between consecutive starts
    notes["dstart"] = df.start.shift(-1) - df.start
    notes = notes.dropna(subset=["dstart"])
    bin_edges = np.linspace(start=0, stop=2, num=n_bins)

    # Create histogram of time differences
    hist, bin_edges = np.histogram(
        notes["dstart"],
        bins=bin_edges,
    )
    distribution = hist.astype(float)

    # Normalize distribution
    total = np.sum(distribution)
    if total > 0:
        distribution = distribution / total

    return distribution


def calculate_dstart_metrics(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    n_bins: int = 50,
) -> dict:
    """
    Calculate correlation coefficient between velocity distributions of two MIDI sequences.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing target notes with columns: pitch, velocity, start, end.
    generated_df : pd.DataFrame
        DataFrame containing generated notes with columns: pitch, velocity, start, end.
    n_bins: int
        Number of bins to quantize dstart values with.

    Returns
    -------
    correlation : float
        Correlation coefficient ranging from -1 to 1.
    metrics : dict
        Additional metrics including dstart distributions.
    """
    target_dist = calculate_dstart_distribution(target_df, n_bins)
    generated_dist = calculate_dstart_distribution(generated_df, n_bins)

    distribution_metrics = calculate_distribution_metrics(
        target_dist=target_dist,
        generated_dist=generated_dist,
    )

    metadata = {
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "target_dstarts": np.count_nonzero(target_dist),
        "generated_dstarts": np.count_nonzero(generated_dist),
    }

    result = {
        "distribution_metrics": distribution_metrics,
        "metadata": metadata,
    }

    return result
