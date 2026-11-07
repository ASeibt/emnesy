"""ABC sampling and relaxed-BP for Sudoku: posteriors from samples or directly from the distributions."""
import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MC_Sampling:
    """problem: SamplingProblem; num_samples: samples per example."""
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

    def get_output(self, evidence, forward=False, sampling=True):
        """evidence (num_evidence_vars, batch_size) -> posteriors (batch_size, num_vars*num_classes),
        with forward=True: predicted evidence (num_evidence_vars, 2, batch_size). sampling=False: relaxed-BP."""
        if not sampling:
            return self.problem.check_evidence(None, evidence.to(self.device), self.batch_size, predict_evidence=forward,
                                               sampling=False, probs=self.probs)

        self.num_evidence_vars = evidence.shape[0]
        samples = self.samples.permute(2, 0, 1)  # (num_samples, batch_size, num_vars)

        if forward:
            evidence_likelihoods = self.problem.check_evidence(samples, evidence.to(self.device), self.batch_size, predict_evidence=True)
            return evidence_likelihoods.permute(1, 2, 0)

        # posterior of the inputs given the evidence
        evidence_checks = self.problem.check_evidence(samples, evidence.to(self.device), self.batch_size)
        # weight of every sampled value: fraction of the constraints of its variable that the sample satisfies
        pair_ids = torch.tensor(self.problem.pair_ids_per_cell, device=self.device).transpose(0, 1)  # (num_pairs_per_cell, num_vars)
        checks = evidence_checks.unsqueeze(-1).expand(self.num_samples, self.batch_size, self.num_evidence_vars, self.num_vars).float()
        index = pair_ids.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.batch_size, pair_ids.shape[0], self.num_vars)
        weights = torch.gather(checks, dim=2, index=index)
        weights = weights.float().mean(dim=2).permute(1, 2, 0)  # (batch_size, num_vars, num_samples)
        posteriors = torch.zeros(self.batch_size, self.num_vars, self.num_classes, device=self.device)
        posteriors.scatter_add_(dim=2, index=self.samples, src=weights)  # sum of the weights per value
        return F.normalize(posteriors.clamp(min=1e-10), p=1, dim=2).reshape(self.batch_size, -1)
