import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import pandas as pd


@dataclass
class SpiralPoint:
    x: float
    y: float
    z: float

    def __add__(
        self,
        other: "SpiralPoint",
    ) -> "SpiralPoint":
        return SpiralPoint(
            self.x + other.x,
            self.y + other.y,
            self.z + other.z,
        )

    def __mul__(
        self,
        scalar: float,
    ) -> "SpiralPoint":
        return SpiralPoint(
            self.x * scalar,
            self.y * scalar,
            self.z * scalar,
        )

    def distance(
        self,
        other: "SpiralPoint",
    ) -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2,
        )


class SpiralArray:
    """
    Implementation of Chew's Spiral Array model for key detection.
    Reference: Chew, E. (2000). Towards a Mathematical Model of Tonality.
    """

    def __init__(self):
        # Parameters for the spiral array
        self.h = math.sqrt(2 / 15)  # height of one rotation
        self.r = 1.0  # radius
        self.p = 4.0  # pitch constant (perfect fifth = 7 semitones)

        # Initialize pitch class representations on the spiral
        self.pitch_classes = self._generate_pitch_spiral()

        # Generate major and minor key representations
        self.major_keys = self._generate_major_keys()
        self.minor_keys = self._generate_minor_keys()

        # Map of key indices to key names
        self.key_names = self._generate_key_names()

    def _generate_pitch_spiral(self) -> Dict[int, SpiralPoint]:
        """Generate two complete spiral arrays (24 points total)"""
        pitch_points = {}
        fifths_steps = {
            0: 0,  # C
            7: 1,  # G
            2: 2,  # D
            9: 3,  # A
            4: 4,  # E
            11: 5,  # B
            6: 6,  # F#
            1: 7,  # C#
            8: 8,  # G#
            3: 9,  # D#
            10: 10,  # A#
            5: 11,  # F
        }

        # First spiral array (0-11)
        for pc in range(12):
            k = fifths_steps[pc]
            theta = k * math.pi / 2
            x = (self.r * 1) * math.cos(theta)
            y = (self.r * 1) * math.sin(theta)
            z = k * self.h
            pitch_points[pc] = SpiralPoint(x, y, z)

        return pitch_points

    def _create_major_chord(self, root: int) -> SpiralPoint:
        third = self.pitch_classes[(root + 4) % 12]
        fifth = self.pitch_classes[(root + 7) % 12]
        chord_point = self.pitch_classes[root % 12] * 0.6 + third * 0.3 + fifth * 0.1
        return chord_point

    def _create_minor_chord(self, root: int) -> SpiralPoint:
        third = self.pitch_classes[(root + 3) % 12]
        fifth = self.pitch_classes[(root + 7) % 12]
        chord_point = self.pitch_classes[root % 12] * 0.5 + third * 0.3 + fifth * 0.2
        return chord_point

    def _generate_major_keys(self) -> List[SpiralPoint]:
        major_keys = []
        for root in range(12):
            tonic = self._create_major_chord(root)
            subdominant = self._create_major_chord((root + 5) % 12)
            dominant = self._create_major_chord((root + 7) % 12)
            major_keys.append(tonic * 0.6 + subdominant * 0.2 + dominant * 0.2)
        return major_keys

    def _generate_minor_keys(self) -> List[SpiralPoint]:
        minor_keys = []
        for root in range(12):
            tonic = self._create_minor_chord(root)  # Tonic
            subdominant = self._create_minor_chord((root + 5) % 12)
            major_dominant = self._create_major_chord((root + 7) % 12)
            minor_dominant = self._create_minor_chord((root + 7) % 12)
            dominant = major_dominant * 0.7 + minor_dominant * 0.3
            minor_keys.append(tonic * 0.5 + subdominant * 0.25 + dominant * 0.25)
        return minor_keys

    def _generate_key_names(self) -> Dict[int, str]:
        """Generate mapping of key indices to key names"""
        pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        key_names = {}

        # Major keys (0-11)
        for i in range(12):
            key_names[i] = f"{pitch_names[i]} major"

        # Minor keys (12-23)
        for i in range(12):
            key_names[i + 12] = f"{pitch_names[i]} minor"

        return key_names

    def get_center_of_effect(
        self,
        pitches: list[int],
        durations: list[float],
        velocities: list[float],
    ) -> SpiralPoint:
        """
        Calculate the center of effect for a set of pitches with durations and velocities,
        selecting the closest pitch on the spiral height-wise.
        """
        if not pitches:
            return SpiralPoint(0, 0, 0)

        # Normalize weights (duration * velocity)
        weights = [d * v for d, v in zip(durations, velocities)]
        total_weight = sum(weights)
        if total_weight == 0:
            return SpiralPoint(0, 0, 0)

        normalized_weights = [w / total_weight for w in weights]

        # Calculate weighted average position
        center = SpiralPoint(0, 0, 0)
        for pitch, weight in zip(pitches, normalized_weights):
            # Find the closest pitch point height-wise for the given pitch
            pitch_class = pitch % 12
            center = center + (self.pitch_classes[pitch_class] * weight)

        return center

    def get_key(
        self,
        center: SpiralPoint,
        return_distribution: bool = False,
    ) -> Union[int, Tuple[int, np.ndarray]]:
        major_distances = [center.distance(k) for k in self.major_keys]
        minor_distances = [center.distance(k) for k in self.minor_keys]
        all_distances = major_distances + minor_distances

        # Enhance the difference in distances for clearer key detection
        exp_distances = [math.exp(-d * 5) for d in all_distances]
        total_exp = sum(exp_distances)
        probabilities = np.array([d / total_exp for d in exp_distances])

        key_index = np.argmin(all_distances)
        if return_distribution:
            return key_index, probabilities
        return key_index


def detect_key_from_notes(
    spiral: SpiralArray,
    notes_df: pd.DataFrame,
    segment_start: float = 0.0,
    segment_duration: Optional[float] = None,
    use_weighted: bool = True,
) -> Tuple[str, np.ndarray]:
    """
    Detect the key from a segment of notes using the Spiral Array model.

    Parameters:
    -----------
    spiral : SpiralArray
        Initialized SpiralArray instance
    notes_df : pd.DataFrame
        DataFrame with columns: pitch, velocity, start, end
    segment_start : float
        Start time of the segment
    segment_duration : Optional[float]
        Duration of the segment. If None, uses the entire notes_df
    use_weighted : bool
        TODO: Explain the weighting mechanism
        Whether to weight notes by duration and velocity

    Returns:
    --------
    key_name : str
        Name of the detected key
    key_probabilities : np.ndarray
        Probability distribution over all possible keys
    """
    if len(notes_df) == 0:
        return "No key detected", np.zeros(24)

    if segment_duration is not None:
        segment_end = segment_start + segment_duration
        mask = (notes_df["start"] < segment_end) & (notes_df["end"] > segment_start)
        segment_df = notes_df[mask].copy()
    else:
        segment_df = notes_df

    pitches = segment_df["pitch"].tolist()

    if use_weighted:
        # Calculate note durations within segment
        durations = []
        for _, note in segment_df.iterrows():
            if segment_duration is not None:
                duration = min(note["end"], segment_end) - max(note["start"], segment_start)
            else:
                duration = note["end"] - note["start"]
            durations.append(duration)
        velocities = [v / 127.0 for v in segment_df["velocity"]]
    else:
        durations = [1.0] * len(pitches)
        velocities = [1.0] * len(pitches)

    center = spiral.get_center_of_effect(pitches, durations, velocities)
    key_index, key_probs = spiral.get_key(center, return_distribution=True)

    return spiral.key_names[key_index], key_probs


def analyze_piece(
    spiral: SpiralArray,
    notes_df: pd.DataFrame,
    segment_duration: float = 0.125,
    use_weighted: bool = True,
) -> Dict:
    if len(notes_df) == 0:
        return {
            "overall_distribution": np.zeros(24),
            "segment_keys": [],
            "top_keys": [],
        }

    total_duration = notes_df["end"].max()
    segments = []
    segment_keys = []
    overall_distribution = np.zeros(24)

    current_time = 0
    while current_time < total_duration:
        key_name, key_probs = detect_key_from_notes(
            spiral=spiral,
            notes_df=notes_df,
            segment_start=current_time,
            segment_duration=segment_duration,
            use_weighted=use_weighted,
        )
        segments.append(key_probs)
        segment_keys.append(key_name)
        overall_distribution += key_probs * segment_duration  # Weight by duration
        current_time += segment_duration

    # Normalize overall distribution
    if np.sum(overall_distribution) > 0:
        overall_distribution = overall_distribution / np.sum(overall_distribution)

    top_key_indices = np.argsort(-overall_distribution)[:3]
    top_keys = [spiral.key_names[i] for i in top_key_indices]

    return {
        "overall_distribution": overall_distribution,
        "segment_keys": segment_keys,
        "top_keys": top_keys,
    }


def calculate_key_metrics(
    target_df: pd.DataFrame,
    generated_df: pd.DataFrame,
    segment_duration: float = 0.5,
    use_weighted: bool = True,
) -> dict:
    """
    Calculate correlation coefficient between target and generated MIDI sequences
    using Spiral Array key detection algorithm.

    Parameters
    ----------
    target_df : pd.DataFrame
        DataFrame containing target notes with columns: pitch, velocity, start, end.
    generated_df : pd.DataFrame
        DataFrame containing generated notes with columns: pitch, velocity, start, end.
    segment_duration : float
        Duration of each segment in seconds for key analysis.
    use_weighted : bool
        If True, weight pitch contributions by note duration and velocity.

    Returns
    -------
    correlation : float
        Correlation coefficient ranging from -1 to 1.
    metrics : dict
        Additional metrics including key distributions.
    """
    spiral = SpiralArray()

    # Analyze both pieces
    target_analysis = analyze_piece(
        spiral=spiral,
        notes_df=target_df,
        segment_duration=segment_duration,
        use_weighted=use_weighted,
    )
    generated_analysis = analyze_piece(spiral, generated_df, segment_duration, use_weighted)

    # Calculate correlation coefficient
    target_dist = target_analysis["overall_distribution"]
    generated_dist = generated_analysis["overall_distribution"]
    correlation = np.corrcoef(target_dist, generated_dist)[0, 1]
    taxicab_distance = np.sum(np.abs(target_dist - generated_dist))

    metrics = {
        "correlation": correlation,
        "taxicab_distance": taxicab_distance,
        "target_distribution": target_dist,
        "generated_distribution": generated_dist,
        "num_segments": len(target_analysis["segment_keys"]),
        "segment_duration": segment_duration,
        "key_names": spiral.key_names,
        "target_top_keys": target_analysis["top_keys"],
        "generated_top_keys": generated_analysis["top_keys"],
    }

    return metrics
