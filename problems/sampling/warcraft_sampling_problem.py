"""Warcraft shortest path for ABC_Sampling: cells are the variables, the evidence is the label path."""
import torch

from inference.solvers.Dijkstra import BackupDijkstra
from problems.sampling.sampling_problem import SamplingProblem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Warcraft_SamplingProblem(SamplingProblem):
    """path_value: extra weight where the sample's path and the label both contain the cell."""
    def __init__(self, dim, num_classes=5, path_value=0):
        self.dim = dim
        self.num_cells = dim**2
        self.num_classes = num_classes
        self.path_value = path_value
        self.dijkstra_solver = BackupDijkstra(grid_size=dim, device=device)
        super().__init__()

    def set_input_var(self):
        return torch.arange(self.num_classes).unsqueeze(0).expand(self.num_cells, self.num_classes)  # every cell can be 0..num_classes-1

    def check_evidence(self, samples, evidence, predict_evidence=False):
        """samples (num_samples, batch_size, num_cells), evidence (batch_size, num_cells).
        Returns per sample and cell 1 (+ path_value on the path) where the paths agree, else 0;
        with predict_evidence: fraction of sampled paths containing each cell (batch_size, num_cells)."""
        num_samples, batch_size = samples.shape[0], samples.shape[1]
        predicted_paths = self.dijkstra_solver(samples.float()).view(num_samples, batch_size, -1)  # 1 on the path
        if predict_evidence:
            return predicted_paths.float().mean(dim=0)

        evidence = evidence.unsqueeze(0).expand(num_samples, -1, -1)
        agrees = (predicted_paths == evidence).float()
        return agrees + agrees * evidence.float() * self.path_value


if __name__ == "__main__":
    # sanity check: python -m problems.sampling.warcraft_sampling_problem
    problem = Warcraft_SamplingProblem(3, path_value=1)
    samples = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0]], [[0, 4, 4, 4, 0, 4, 4, 4, 0]]])  # (num_samples=2, batch_size=1, 9)
    print("paths:", problem.check_evidence(samples, evidence=torch.zeros(1, 9), predict_evidence=True))  # diagonal for both maps
    print("weights:", problem.check_evidence(samples, evidence=torch.tensor([[1, 0, 0, 0, 1, 0, 0, 0, 1]])))
