from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Any
import pandas as pd
from piano_metrics.f1_piano import calculate_f1
from piano_metrics.key_distribution import calculate_key_correlation

@dataclass
class MetricResult:
    """Store metric calculation results with metadata"""
    value: float
    metadata: Dict[str, Any] = None

class PianoMetric(ABC):
    """Abstract base class for all piano performance metrics"""
    
    @abstractmethod
    def calculate(self, target_df: pd.DataFrame, generated_df: pd.DataFrame) -> MetricResult:
        """Calculate metric between target and generated performances"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric identifier"""
        pass

class F1Metric(PianoMetric):
    """Wrapper for F1 score calculation"""
    
    def __init__(self, use_pitch_class: bool = False, velocity_threshold: float = 30, min_time_unit: float = 0.01):
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
    
    def calculate(self, target_df: pd.DataFrame, generated_df: pd.DataFrame) -> MetricResult:
        f1_score, detailed_metrics = calculate_f1(
            target_df=target_df,
            generated_df=generated_df,
            velocity_threshold=self.velocity_threshold,
            use_pitch_class=self.use_pitch_class
        )
        return MetricResult(
            value=f1_score,
            metadata={
                "detailed_metrics": detailed_metrics,
                "config": self.config,
            }
        )
    
    @property
    def name(self) -> str:
        return f"f1{'_pitch_class' if self.use_pitch_class else ''}"

class KeyCorrelationMetric(PianoMetric):
    """Wrapper for key correlation calculation"""
    
    def __init__(self, segment_duration: float = 0.125, use_weighted: bool = True):
        self.segment_duration = segment_duration
        self.use_weighted = use_weighted

    @property
    def config(self):
        return {
            "segment_duration": self.segment_duration,
            "use_weighted": self.use_weighted,
        }
    
    def calculate(self, target_df: pd.DataFrame, generated_df: pd.DataFrame) -> MetricResult:
        correlation, metrics = calculate_key_correlation(
            target_df=target_df,
            generated_df=generated_df,
            segment_duration=self.segment_duration,
            use_weighted=self.use_weighted
        )
        return MetricResult(value=correlation, metadata={"detailed_metrics": metrics},)
    
    @property
    def name(self) -> str:
        return f"key_correlation{'_weighted' if self.use_weighted else '_unweighted'}"

class MetricsRunner:
    """Orchestrates the calculation of multiple metrics"""
    
    def __init__(self, metrics: List[PianoMetric]):
        self.metrics = {}
        for metric in metrics:
            self.register_metric(metric)
        
    
    def calculate_all(self, target_df: pd.DataFrame, generated_df: pd.DataFrame) -> Dict[str, MetricResult]:
        """Calculate all metrics for a single example"""
        results = {}
        for metric_name, metric in self.metrics.items():
            results[metric_name] = metric.calculate(target_df, generated_df)
        return results
    
    def calculate_batch(self, targets: List[pd.DataFrame], generations: List[pd.DataFrame]) -> Dict[str, List[MetricResult]]:
        """Calculate metrics for a batch of examples"""
        batch_results = {metric_name: [] for metric_name in self.metrics}
        
        for target_df, generated_df in zip(targets, generations):
            results = self.calculate_all(target_df, generated_df)
            for metric_name, result in results.items():
                batch_results[metric_name].append(result)
                
        return batch_results
    
    def register_metric(self, metric: PianoMetric):
        if not hasattr(metric, "name") or not metric.name:
            raise ValueError("Task class must have a 'name' attribute.")

        if metric.name in self.metrics:
            raise ValueError(f"Metric '{metric.name}' is already registered.")
        self.metrics[metric.name] = metric

    def list_metrics(self) -> list[str]:
        return list(self.metrics.keys())
