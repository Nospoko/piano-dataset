from typing import Dict, Tuple

import numpy as np
import pandas as pd


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
    notes["dstart"] = df.start - df.start.shift(-1)
    notes = notes.dropna(subset=["dstart"])
    
    # Create histogram of time differences
    hist, bin_edges = np.histogram(
        notes["dstart"],
        bins=np.linspace(start=0, stop=2, num=n_bins), 
        range=(0, notes["dstart"].quantile(0.99)), 
    )
    distribution = hist.astype(float)
    
    # Normalize distribution
    total = np.sum(distribution)
    if total > 0:
        distribution = distribution / total
        
    return distribution


def calculate_dstart_correlation(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    n_bins: int = 50,
) -> Tuple[float, Dict]:
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
        Additional metrics including pitch distributions.
    """
    target_dist = calculate_dstart_distribution(target_df, n_bins)
    generated_dist = calculate_dstart_distribution(generated_df, n_bins)

    correlation = np.corrcoef(target_dist, generated_dist)[0, 1]

    metrics = {
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "target_active_pitches": np.count_nonzero(target_dist),
        "generated_active_pitches": np.count_nonzero(generated_dist),
        "pitch_range": list(range(21, 109)),  # MIDI pitch numbers
    }

    return correlation, metrics
