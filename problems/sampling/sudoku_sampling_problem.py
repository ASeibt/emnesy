"""Sudoku for MC_Sampling: cells are the variables, cells in a row, column or box must differ."""
from itertools import combinations

import torch
import torch.nn.functional as F

from problems.sampling.sampling_problem import SamplingProblem

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sudoku_SamplingProblem(SamplingProblem):
    def __init__(self, dimension):  # 2 for 4x4, 3 for 9x9, ...
        self.dimension = dimension
        self.num_digits = dimension**2
        self.num_cells = dimension**4
        super().__init__()
        self.all_pairs = self.determine_all_pairs()  # (num_pairs, 2)
        self.pair_ids_tensor = torch.tensor(self.other_cells_per_cell, dtype=torch.long, device=device)  # (num_cells, num_pairs_per_cell)

    def determine_all_pairs(self):
        """Pairs of cells that must differ (num_pairs, 2). Also sets per cell its pair ids (ABC sampling)
        and the other cells of these pairs (relaxed-BP)."""
        cell_ids = torch.arange(self.num_cells).reshape(self.num_digits, self.num_digits)
        n = self.dimension
        rows = [cell_ids[i].tolist() for i in range(self.num_digits)]
        cols = [cell_ids[:, i].tolist() for i in range(self.num_digits)]
        boxes = [cell_ids[i*n:(i+1)*n, j*n:(j+1)*n].flatten().tolist() for i in range(n) for j in range(n)]
        all_pair_list = [pair for group in rows + cols + boxes for pair in combinations(group, 2)]
        all_pair_set = set(tuple(sorted(pair)) for pair in all_pair_list)
        all_pair_list = list(all_pair_set)

        self.pair_ids_per_cell = [[] for _ in range(self.num_cells)]
        self.other_cells_per_cell = [[] for _ in range(self.num_cells)]
        for pair in all_pair_set:
            pair_id = all_pair_list.index(pair)
            self.pair_ids_per_cell[pair[0]].append(pair_id)
            self.pair_ids_per_cell[pair[1]].append(pair_id)
            self.other_cells_per_cell[pair[0]].append(pair[1])
            self.other_cells_per_cell[pair[1]].append(pair[0])
        return torch.tensor(all_pair_list, device=device)

    def set_input_var(self):
        return torch.arange(self.num_digits).unsqueeze(0).expand(self.num_cells, self.num_digits)  # every cell can be 0..num_digits-1

    def check_evidence(self, samples, evidence, batch_size=None, predict_evidence=False, sampling=True, probs=None):
        """sampling=True: per sample and pair, whether it matches the evidence (num_samples, batch_size, num_pairs);
        with predict_evidence: fraction of samples with equal / different cells (batch_size, num_pairs, 2).
        sampling=False (relaxed-BP): probs * (1 - mean probs of the other cells), normalized (batch_size, num_cells*num_digits)."""
        if sampling:
            different = samples[:, :, self.all_pairs[:, 0]] != samples[:, :, self.all_pairs[:, 1]]  # (num_samples, batch_size, num_pairs)
            if predict_evidence:
                return F.one_hot(different.to(torch.long), num_classes=2).float().mean(dim=0)
            return different == evidence.transpose(0, 1).unsqueeze(0)

        if predict_evidence:
            raise ValueError("relaxed-BP (sampling=False) only computes posteriors, not predicted evidence")
        idx = self.pair_ids_tensor  # (num_cells, num_pairs_per_cell)
        probs_expanded = probs.unsqueeze(1).expand(batch_size, self.num_cells, self.num_cells, self.num_digits)
        idx_expanded = idx.unsqueeze(0).unsqueeze(-1).expand(batch_size, self.num_cells, idx.shape[1], self.num_digits)
        probs_others = torch.gather(probs_expanded, dim=2, index=idx_expanded)
        posteriors = F.normalize(probs * (1 - probs_others.float().mean(dim=2)), p=1, dim=-1)
        return posteriors.reshape(batch_size, -1)


if __name__ == "__main__":
    # sanity check: python -m problems.sampling.sudoku_sampling_problem
    problem = Sudoku_SamplingProblem(dimension=2)
    num_pairs = problem.all_pairs.shape[0]
    valid = [0, 1, 2, 3, 2, 3, 0, 1, 1, 0, 3, 2, 3, 2, 1, 0]
    samples = torch.tensor([[valid], [[0] * 16]], device=device)
    checks = problem.check_evidence(samples, evidence=torch.ones(num_pairs, 1, device=device))
    print(f"{num_pairs} pairs; satisfied in valid grid: {checks[0].all().item()}, in all-zero grid: {checks[1].any().item()}")
