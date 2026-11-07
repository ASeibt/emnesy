"""ABC sampling for Warcraft: weight sampled cell values by whether their shortest path agrees with the label."""
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ABC_Sampling:
    """problem: Warcraft_SamplingProblem; num_samples: samples per example."""
    def __init__(self, problem, batch_size, num_samples, device=device):
        self.problem = problem
        self.num_vars = problem.input_var.shape[0]
        self.num_classes = problem.input_var.shape[1]
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.device = device

    def set_input(self, probs):
        """probs: (batch_size, num_vars, num_classes)"""
        if probs.shape != (self.batch_size, self.num_vars, self.num_classes):
            raise ValueError(f"probs shape should be ({self.batch_size}, {self.num_vars}, {self.num_classes}) but found {probs.shape}")
        self.probs = probs.to(self.device)

    def sample(self):
        """self.samples: (batch_size, num_vars, num_samples)"""
        flat_probs = self.probs.reshape(self.batch_size * self.num_vars, self.num_classes)
        self.samples = torch.multinomial(flat_probs, num_samples=self.num_samples, replacement=True).reshape(
            self.batch_size, self.num_vars, self.num_samples)

    def get_output(self, evidence, forward=False):
        """evidence: label path (batch_size, num_vars) -> posteriors (batch_size, num_vars, num_classes);
        forward=True: fraction of samples whose path contains each cell (num_vars, batch_size)."""
        self.num_evidence_vars = evidence.shape[1]
        samples = self.samples.permute(2, 0, 1)  # (num_samples, batch_size, num_vars)

        if forward:
            average_paths = self.problem.check_evidence(samples, evidence.to(self.device), predict_evidence=True)
            return average_paths.transpose(0, 1)

        weights = self.problem.check_evidence(samples, evidence.to(self.device))  # (num_samples, batch_size, num_vars)
        posteriors = torch.zeros(self.batch_size, self.num_vars, self.num_classes, device=self.device).clamp(min=1e-10)
        posteriors.scatter_add_(dim=2, index=self.samples, src=weights.float().permute(1, 2, 0))  # sum of the weights per value
        return F.normalize(posteriors, p=1, dim=2, eps=1e-10)
