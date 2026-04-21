from __future__ import annotations

import random
from abc import ABC, abstractmethod
from typing import Any

BENCHMARK_REGISTRY: dict[str, Benchmark] = {}


class Benchmark(ABC):
    """Base class for evaluation benchmarks."""

    name: str
    requires_judge: bool = False
    max_new_tokens: int = 512
    repetition_penalty: float = 1.0

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "name") and cls.name:
            BENCHMARK_REGISTRY[cls.name] = cls()

    @abstractmethod
    def load_all(self) -> list[dict]:
        """Load all samples from the benchmark."""

    def load_subset(self, n: int, seed: int) -> list[dict]:
        """Load n random samples with a fixed seed for reproducibility."""
        all_samples = self.load_all()
        rng = random.Random(seed)
        n = min(n, len(all_samples))
        return rng.sample(all_samples, n)

    @abstractmethod
    def format_prompt(self, sample: dict) -> dict:
        """Format a sample into model input (video path + conversation).

        Returns dict with:
            video: str (path to video)
            prompt: str (user prompt text)
        """

    @abstractmethod
    def score(self, predictions: list[dict], judge: Any = None) -> dict[str, float]:
        """Score predictions against ground truth.

        Args:
            predictions: list of {sample: dict, prediction: str}
            judge: optional GPT-4o judge wrapper for judge-based metrics

        Returns:
            {metric_name: value} dict
        """

    def metric_names(self) -> list[str]:
        """Return list of metric names this benchmark produces."""
        return []
