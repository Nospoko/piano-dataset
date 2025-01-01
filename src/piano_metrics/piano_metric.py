from dataclasses import dataclass
from abc import ABC, abstractmethod
import importlib.resources as pkg_resources

import yaml
import pandas as pd

from piano_metrics.f1_piano import calculate_f1
from piano_metrics.key_distribution import calculate_key_metrics
from piano_metrics.pitch_distribution import calculate_pitch_metrics
from piano_metrics.dstart_distribution import calculate_dstart_metrics
from piano_metrics.duration_distribution import calculate_duration_metrics
from piano_metrics.velocity_distribution import calculate_velocity_metrics


@dataclass
class MetricResult:
    """Store metric calculation results with metadata"""

    value: float
    metadata: dict
    metric_config: dict


class PianoMetric(ABC):
    """Abstract base class for all piano performance metrics"""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def calculate(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> MetricResult:
        """Calculate metric between target and generated performances"""
        pass


class F1Metric(PianoMetric):
    """Wrapper for F1 score calculation"""

    def __init__(
        self,
        name: str,
        use_pitch_class: bool = False,
        velocity_threshold: float = 30,
        min_time_unit: float = 0.01,
    ):
        super().__init__(name=name)
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
            value=f1_metrics["weighted_f1"],
            metadata=f1_metrics,
            metric_config=self.config,
        )
        return result


class KeyCorrelationMetric(PianoMetric):
    """Wrapper for key correlation calculation"""

    def __init__(
        self,
        name: str,
        segment_duration: float = 0.125,
        use_weighted: bool = True,
    ):
        super().__init__(name=name)
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
        key_metrics = calculate_key_metrics(
            target_df=target_df,
            generated_df=generated_df,
            segment_duration=self.segment_duration,
            use_weighted=self.use_weighted,
        )
        result = MetricResult(
            value=key_metrics["taxicab_distance"],
            metadata=key_metrics,
            metric_config=self.config,
        )
        return result


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
        dstart_metrics = calculate_dstart_metrics(
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
        duration_metrics = calculate_duration_metrics(
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
        pitch_metrics = calculate_pitch_metrics(
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


class MetricFactory:
    _registry = {
        "F1Metric": F1Metric,
        "KeyCorrelationMetric": KeyCorrelationMetric,
    }

    @classmethod
    def create_metric(
        cls,
        metric_name: str,
        class_name: str,
        params: dict,
    ) -> PianoMetric:
        if class_name not in cls._registry:
            raise ValueError(f"Unknown metric class: {class_name}")
        metric = cls._registry[class_name](
            name=metric_name,
            **params,
        )
        return metric


class MetricsRunner:
    """Orchestrates the calculation of multiple metrics"""

    def __init__(self, metrics_config: list[dict]):
        self.metrics_config = metrics_config
        self.metrics = []

        for metric_config in metrics_config:
            print(metric_config)
            piano_metric = MetricFactory.create_metric(
                class_name=metric_config["class"],
                metric_name=metric_config["name"],
                params=metric_config["params"],
            )
            self.metrics.append(piano_metric)

    @classmethod
    def load_default(cls) -> "MetricsRunner":
        with pkg_resources.open_text("configs", "metrics-default.yaml") as f:
            metrics_config = yaml.safe_load(f)

        this = cls(metrics_config)
        return this

    def calculate_all(
        self,
        target_df: pd.DataFrame,
        generated_df: pd.DataFrame,
    ) -> dict[str, MetricResult]:
        """Calculate all metrics for a single example"""
        results = {}
        for piano_metric in self.metrics:
            results[piano_metric.name] = piano_metric.calculate(
                target_df=target_df,
                generated_df=generated_df,
            )
        return results

    def calculate_batch(
        self,
        targets: list[pd.DataFrame],
        generations: list[pd.DataFrame],
    ) -> dict[str, list[MetricResult]]:
        """Calculate metrics for a batch of examples"""
        batch_results = {piano_metric.name: [] for piano_metric in self.metrics}

        for target_df, generated_df in zip(targets, generations):
            results = self.calculate_all(target_df, generated_df)
            # FIXME This whole method should probably be left for the user to implement (i.e. removed)
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
