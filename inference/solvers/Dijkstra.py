# -*- coding: utf-8 -*-
# Dijkstra shortest paths for Warcraft, parallelised with ray; mostly from https://github.com/uclnlp/torch-imle

import heapq
import itertools
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import einops as E
import numpy as np
import ray
import torch
from torch import Tensor
from torch.nn import Module

DijkstraOutput = namedtuple(
    "DijkstraOutput", ["shortest_path", "is_unique", "transitions"]
)

# Weight mapping: category index → terrain traversal cost
_CATEGORY_WEIGHTS = np.array([0.8, 1.2, 5.3, 7.7, 9.2], dtype=np.float32)

# Start of code taken from https://github.com/uclnlp/torch-imle


def neighbours_8(x, y, x_max, y_max):
    deltas_x = (-1, 0, 1)
    deltas_y = (-1, 0, 1)
    for dx, dy in itertools.product(deltas_x, deltas_y):
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def neighbours_4(x, y, x_max, y_max):
    for dx, dy in [(1, 0), (0, 1), (0, -1), (-1, 0)]:
        x_new, y_new = x + dx, y + dy
        if 0 <= x_new < x_max and 0 <= y_new < y_max and (dx, dy) != (0, 0):
            yield x_new, y_new


def get_neighbourhood_func(neighbourhood_fn):
    if neighbourhood_fn == "4-grid":
        return neighbours_4
    elif neighbourhood_fn == "8-grid":
        return neighbours_8
    else:
        raise Exception(f"neighbourhood_fn of {neighbourhood_fn} not possible")


def dijkstra(matrix, neighbourhood_fn="8-grid", request_transitions=False): # matrix shape (H, W)
    x_max, y_max = matrix.shape
    neighbors_func = partial(
        get_neighbourhood_func(neighbourhood_fn), x_max=x_max, y_max=y_max
    )

    costs = np.full_like(matrix, 1.0e10)
    costs[0][0] = matrix[0][0]
    num_path = np.zeros_like(matrix)
    num_path[0][0] = 1
    priority_queue = [(matrix[0][0], (0, 0))]
    certain = set()
    transitions = dict()

    while priority_queue:
        cur_cost, (cur_x, cur_y) = heapq.heappop(priority_queue)
        if (cur_x, cur_y) in certain:
            pass

        for x, y in neighbors_func(cur_x, cur_y):
            if (x, y) not in certain:
                if matrix[x][y] + costs[cur_x][cur_y] < costs[x][y]:
                    costs[x][y] = matrix[x][y] + costs[cur_x][cur_y]
                    heapq.heappush(priority_queue, (costs[x][y], (x, y)))
                    transitions[(x, y)] = (cur_x, cur_y)
                    num_path[x, y] = num_path[cur_x, cur_y]
                elif matrix[x][y] + costs[cur_x][cur_y] == costs[x][y]:
                    num_path[x, y] += 1

        certain.add((cur_x, cur_y))
    # retrieve the path
    cur_x, cur_y = x_max - 1, y_max - 1
    on_path = np.zeros_like(matrix)
    on_path[-1][-1] = 1
    while (cur_x, cur_y) != (0, 0):
        cur_x, cur_y = transitions[(cur_x, cur_y)]
        on_path[cur_x, cur_y] = 1.0

    is_unique = num_path[-1, -1] == 1

    if request_transitions:
        return DijkstraOutput(
            shortest_path=on_path, is_unique=is_unique, transitions=transitions
        )
    else:
        return DijkstraOutput(
            shortest_path=on_path, is_unique=is_unique, transitions=None
        )


def get_solver(neighbourhood_fn, request_transitions):
    def solver(matrix):
        return dijkstra(matrix, neighbourhood_fn, request_transitions).shortest_path

    return solver


def maybe_parallelize(function, arg_list):
    if ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]


def compute_shortest_path(
    batch_weights_BHW: torch.Tensor,
    neighbourhood_fn="8-grid",
    request_transitions=False,
    debug=False,
):
    solver = get_solver(neighbourhood_fn, request_transitions)
    weights = batch_weights_BHW.detach().cpu().numpy()
    if debug:
        start = time.time()
    shortest_paths = np.asarray(maybe_parallelize(solver, arg_list=list(weights)))
    if debug:
        end = time.time()
        print(f"Number of paths: {len(weights)}")
        print(f"Time taken: {end - start} seconds")
    shortest_paths = (
        torch.from_numpy(shortest_paths).float().to(batch_weights_BHW.device)
    )
    return shortest_paths


# End of code taken from https://github.com/uclnlp/torch-imle


class BackupDijkstra(Module):
    def __init__(self, grid_size: int, device: torch.device):
        super().__init__()
        self.grid_size = grid_size
        self.device = device

    def forward(self, theta: Tensor) -> Tensor: # shape (S, B, G1*G2)- sample dimension, batch dimension, flattened grid dimension
        s = theta.size(0)
        b = theta.size(1)

        weights = E.rearrange(
            theta, "s b (d1 d2) -> (s b) d1 d2", d1=self.grid_size, d2=self.grid_size
        )
        weights = torch.where(weights == 0, 0.8, weights)
        weights = torch.where(weights == 1, 1.2, weights)
        weights = torch.where(weights == 2, 5.3, weights)
        weights = torch.where(weights == 3, 7.7, weights)
        weights = torch.where(weights == 4, 9.2, weights)

        best_path = compute_shortest_path(weights)
        best_path = E.rearrange(best_path, "(s b) d1 d2 -> s b d1 d2", s=s, b=b)
        return best_path # shape (S, B, G1, G2) - binary tensor with 1 where on the path and 0 elsewhere
