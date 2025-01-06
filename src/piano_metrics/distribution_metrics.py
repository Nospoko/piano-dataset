import numpy as np
from scipy.stats import wasserstein_distance


def calculate_distribution_metrics(
    target_dist: np.array,
    generated_dist: np.array,
) -> dict:
    correlation = np.corrcoef(target_dist, generated_dist)[0, 1]
    taxicab_distance = np.sum(np.abs(target_dist - generated_dist))
    wasserstein = wasserstein_distance(
        u_values=target_dist,
        v_values=generated_dist,
    )

    metrics = {
        "correlation": correlation,
        "taxicab_distance": taxicab_distance,
        "wasserstein": wasserstein,
    }
    return metrics
