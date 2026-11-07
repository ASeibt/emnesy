"""Base class of the sampling problems."""
from abc import ABC, abstractmethod


class SamplingProblem(ABC):
    def __init__(self):
        self.input_var = self.set_input_var()

    @abstractmethod
    def set_input_var(self):
        """Possible values per variable (num_vars, num_classes)."""

    @abstractmethod
    def check_evidence(self, samples, evidence, batch_size, predict_evidence=False):
        """Check samples (num_samples, batch_size, num_vars) against the evidence."""
