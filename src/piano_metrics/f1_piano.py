import pandas as pd


def normalize_pitch_to_class(pitch: int) -> int:
    """Convert MIDI pitch (0-127) to pitch class (0-11)"""
    return pitch % 12


def calculate_f1(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    min_time_unit: float = 0.01,
    velocity_threshold: float = 30,
    use_pitch_class: bool = True,
) -> dict:
    """
    Calculate F1 score between target and generated MIDI-like note sequences.
    Only calculates at note boundary events and weights by duration.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing target notes with columns: pitch, velocity, start, end.
    generated_df : pd.DataFrame
        DataFrame containing generated notes with columns: pitch, velocity, start, end.
    min_time_step : float
        Minimum time unit for duration calculations (in seconds).
    velocity_threshold : float
        Maximum allowed velocity difference for notes to be considered matching.
    use_pitch_class : bool
        If True, normalize pitches to pitch classes (0-11), treating octaves as equivalent.

    Returns
    -------
    f1_score : float
        Duration-weighted average F1 score.
    metrics : dict
        Detailed metrics including precision and recall per event.
    """
    # Get all unique time points where notes change (starts or ends)
    target_time_points = list(target_df["start"]) + list(target_df["end"])
    generated_time_points = list(generated_df["start"]) + list(generated_df["end"])
    time_points = sorted(set(target_time_points + generated_time_points))

    if not time_points:
        return 0.0, {
            "time_points": [],
            "precision": [],
            "recall": [],
            "f1": [],
            "durations": [],
        }

    metrics = {
        "time_points": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "durations": [],  # store duration of each segment in minimum time units
    }

    def get_active_notes(
        df: pd.DataFrame,
        time_point: float,
    ) -> set[tuple]:
        """
        Return set of (pitch, velocity) tuples active at given time point.
        If use_pitch_class is True, normalize pitches to 0-11.
        """
        mask = (df["start"] <= time_point) & (df["end"] >= time_point)
        return set(zip(df[mask]["pitch"], df[mask]["velocity"]))

    def find_matching_notes(
        target_notes: set[tuple],
        generated_notes: set[tuple],
        velocity_threshold: float,
    ) -> int:
        """Count matching notes, ensuring each generated note matches at most one target note"""
        matches = 0
        used_generated = set()

        if use_pitch_class:
            target_notes = {(normalize_pitch_to_class(pitch), vel) for pitch, vel in target_notes}
            generated_notes = {(normalize_pitch_to_class(pitch), vel) for pitch, vel in generated_notes}

        for t_pitch, t_vel in target_notes:
            for g_pitch, g_vel in generated_notes:
                was_used = (g_pitch, g_vel) in used_generated
                is_pitch_equal = t_pitch == g_pitch
                within_velocity_threshold = abs(t_vel - g_vel) <= velocity_threshold
                if not was_used and is_pitch_equal and within_velocity_threshold:
                    matches += 1
                    used_generated.add((g_pitch, g_vel))
                    break

        return matches

    # Calculate metrics for each segment between events
    for i in range(len(time_points) - 1):
        current_time = time_points[i]
        next_time = time_points[i + 1]
        duration = next_time - current_time

        # Convert duration to number of minimum time units
        duration_units = round(duration / min_time_unit)

        target_notes = get_active_notes(
            df=target_df,
            time_point=current_time + min_time_unit / 2,
        )
        generated_notes = get_active_notes(
            df=generated_df,
            time_point=current_time + min_time_unit / 2,
        )

        if len(target_notes) == 0 and len(generated_notes) == 0:
            continue

        true_positives = find_matching_notes(
            target_notes=target_notes,
            generated_notes=generated_notes,
            velocity_threshold=velocity_threshold,
        )

        precision = true_positives / len(generated_notes) if len(generated_notes) > 0 else 0
        recall = true_positives / len(target_notes) if len(target_notes) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics["time_points"].append(current_time)
        metrics["precision"].append(precision)
        metrics["recall"].append(recall)
        metrics["f1"].append(f1)
        metrics["durations"].append(duration_units)

    # Calculate duration-weighted average F1 score
    total_duration = sum(metrics["durations"])
    f1_sum = sum(f1 * dur for f1, dur in zip(metrics["f1"], metrics["durations"]))
    weighted_f1 = f1_sum / total_duration if total_duration > 0 else 0.0

    metrics["f1_sum"] = f1_sum
    metrics["weighted_f1"] = weighted_f1

    return metrics
