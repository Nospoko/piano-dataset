from dataclasses import dataclass
from typing import Any, Dict, List
from abc import ABC, abstractmethod

import pandas as pd

from piano_metrics.f1_piano import calculate_f1
from piano_metrics.key_distribution import calculate_key_correlation
from piano_metrics.pitch_distribution import calculate_pitch_correlation
from piano_metrics.dstart_distribution import calculate_dstart_correlation
from piano_metrics.velocity_distribution import calculate_velocity_metrics
from piano_metrics.duration_distribution import calculate_duration_correlation


@dataclass
class MetricResult:
    """Store metric calculation results with metadata"""

    metics: dict
    metadata: Dict[str, Any] = None


class PianoMetric(ABC):
    """Abstract base class for all piano performance metrics"""

    @abstractmethod
    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        """Calculate metric between target and generated performances"""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric identifier"""
        pass


class F1Metric(PianoMetric):
    """Wrapper for F1 score calculation"""

    def __init__(
        self,
        use_pitch_class: bool = False,
        velocity_threshold: float = 30,
        min_time_unit: float = 0.01,
    ):
        self.use_pitch_class = use_pitch_class
        self.velocity_threshold = velocity_threshold
        self.min_time_unit = min_time_unit

    @property
    def config(self):
        return {
            "use_pitch_class": self.use_pitch_class,
            "velocity_threshold": self.velocity_threshold,
            "min_time_unit": self.min_time_unit,
        }

    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        f1_metrics = calculate_f1(
            target_df=target_df,
            generated_df=generated_df,
            velocity_threshold=self.velocity_threshold,
            use_pitch_class=self.use_pitch_class,
        )
        result = MetricResult(
            metrics=f1_metrics,
            metadata={
                "config": self.config,
            },
        )
        return result

    @property
    def name(self) -> str:
        return f"f1{'_pitch_class' if self.use_pitch_class else ''}"


class KeyCorrelationMetric(PianoMetric):
    """Wrapper for key correlation calculation"""

    def __init__(
        self,
        segment_duration: float = 0.125,
        use_weighted: bool = True,
    ):
        self.segment_duration = segment_duration
        self.use_weighted = use_weighted

    @property
    def config(self):
        return {
            "segment_duration": self.segment_duration,
            "use_weighted": self.use_weighted,
        }

    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        key_metrics = calculate_key_correlation(
            target_df=target_df,
            generated_df=generated_df,
            segment_duration=self.segment_duration,
            use_weighted=self.use_weighted,
        )
        result = MetricResult(metrics=key_metrics)
        return result

    @property
    def name(self) -> str:
        return f"key_correlation{'_weighted' if self.use_weighted else '_unweighted'}"


class DstartCorrelationMetric(PianoMetric):
    """Wrapper for dstart (time between consecutive notes) correlation calculation"""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins

    @property
    def config(self):
        return {
            "n_bins": self.n_bins,
        }

    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        dstart_metrics = calculate_dstart_correlation(
            target_df=target_df,
            generated_df=generated_df,
            n_bins=self.n_bins,
        )
        result = MetricResult(
            metrics=dstart_metrics,
            metadata={
                "config": self.config,
            },
        )
        return result

    @property
    def name(self) -> str:
        return "dstart_correlation"


class DurationCorrelationMetric(PianoMetric):
    """Wrapper for note duration correlation calculation"""

    def __init__(self, n_bins: int = 50):
        self.n_bins = n_bins

    @property
    def config(self):
        return {
            "n_bins": self.n_bins,
        }

    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        duration_metrics = calculate_duration_correlation(
            target_df=target_df,
            generated_df=generated_df,
            n_bins=self.n_bins,
        )
        result = MetricResult(
            mtrics=duration_metrics,
            metadata={
                "config": self.config,
            },
        )
        return result

    @property
    def name(self) -> str:
        return "duration_correlation"


class VelocityCorrelationMetric(PianoMetric):
    """Wrapper for velocity distribution correlation calculation"""

    def __init__(self, use_weighted: bool = True):
        self.use_weighted = use_weighted

    @property
    def config(self):
        return {
            "use_weighted": self.use_weighted,
        }

    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        velocity_metrics = calculate_velocity_metrics(
            target_df=target_df,
            generated_df=generated_df,
            use_weighted=self.use_weighted,
        )
        result = MetricResult(
            mtrics=velocity_metrics,
            metadata={
                "config": self.config,
            },
        )
        return result

    @property
    def name(self) -> str:
        return f"velocity_correlation{'_weighted' if self.use_weighted else '_unweighted'}"


class PitchCorrelationMetric(PianoMetric):
    """Wrapper for pitch distribution correlation calculation"""

    def __init__(self, use_weighted: bool = True):
        self.use_weighted = use_weighted

    @property
    def config(self):
        return {
            "use_weighted": self.use_weighted,
        }

    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        pitch_metrics = calculate_pitch_correlation(
            target_df=target_df,
            generated_df=generated_df,
            use_weighted=self.use_weighted,
        )
        result = MetricResult(
            metrics=pitch_metrics,
            metadata={
                "config": self.config,
            },
        )
        return result

    @property
    def name(self) -> str:
        return f"pitch_correlation{'_weighted' if self.use_weighted else '_unweighted'}"


class MetricsRunner:
    """Orchestrates the calculation of multiple metrics"""

    def __init__(self, metrics: List[PianoMetric]):
        self.metrics = {}
        for metric in metrics:
            self.register_metric(metric)

    def calculate_all(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> Dict[str, MetricResult]:
        """Calculate all metrics for a single example"""
        results = {}
        for metric_name, metric in self.metrics.items():
            try:
                results[metric_name] = metric.calculate(target_df, generated_df)
            except Exception as e:
                print(f"Error when calculating metric {metric_name}: {e}")
                results[metric_name] = MetricResult(0.0, None)
        return results

    def calculate_batch(
        self,
        targets: List[pd.DataFrame],
        generations: List[pd.DataFrame],
    ) -> Dict[str, List[MetricResult]]:
        """Calculate metrics for a batch of examples"""
        batch_results = {metric_name: [] for metric_name in self.metrics}

        for target_df, generated_df in zip(targets, generations):
            results = self.calculate_all(target_df, generated_df)
            for metric_name, result in results.items():
                batch_results[metric_name].append(result)

        return batch_results

    def register_metric(self, metric: PianoMetric):
        # TODO This should be checked in the PianoMetric ABC implementation
        if not hasattr(metric, "name") or not metric.name:
            raise ValueError("Task class must have a 'name' attribute.")

        if metric.name in self.metrics:
            raise ValueError(f"Metric '{metric.name}' is already registered.")
        self.metrics[metric.name] = metric

    def list_metrics(self) -> list[str]:
        return list(self.metrics.keys())
