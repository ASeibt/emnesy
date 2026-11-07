""".uai factor graphs for Sudoku. Run from the repo root with the dimensions (2: 4x4, 3: 9x9; default 2 to 6).

Variables: cells, one check per pair of cells that must differ, a chain of ANDs ending in 'sudoku valid';
state 0 is 'true'. First line of a file: last check variable id (excl.), output factor id, first and last
(excl.) prior factor id."""
import sys
from itertools import combinations

import torch


def generate_uai_sudoku(dimension):  # 2 for 4x4, 3 for 9x9, ...
    ## def variables
    n = dimension
    num_cells = n**4 # number of cells
    num_row_pairs = n**2 * n**2 * (n**2 -1) // 2
    num_box_pairs = n**4 * ((n-1)**2) // 2

    # determine all variable pairs that need to be checked for not-equal constraint
    input_var_ids = torch.arange(num_cells).reshape(n**2, n**2)  # shape: (n^2, n^2)
    rows = [input_var_ids[i].tolist() for i in range(n**2)]
    cols = [input_var_ids[:, i].tolist() for i in range(n**2)]
    boxes = [input_var_ids[i*n:(i+1)*n, j*n:(j+1)*n].flatten().tolist() for i in range(n) for j in range(n)]
    row_pairs = [list(combinations(row, 2)) for row in rows]
    col_pairs = [list(combinations(col, 2)) for col in cols]
    box_pairs = [list(combinations(box, 2)) for box in boxes]
    all_pair_list = [pair for sublist in row_pairs + col_pairs + box_pairs for pair in sublist] # list of tuples
    all_pairs = set(tuple(sorted(pair)) for pair in all_pair_list)
    num_checks = len(all_pairs)

    var_dims = [n**2] * num_cells + [2] * (2*num_checks - 1)

    ###############################################################################################
    ## def factors

    # initialize lists for factor specs + values
    factor_specs = []
    factor_values = []

    # input factors
    for i in range(num_cells):
        factor_specs.append([i])
        factor_values.append([1/(n**2)] * (n**2)) # uniform distribution

    # not-equal tensor
    not_equal_tensor = torch.stack((torch.ones(n**2, n**2) - torch.eye(n**2),torch.zeros(n**2, n**2) + torch.eye(n**2)), dim=2)

    # check factors
    check_var_id = num_cells # first check var id
    for pair in all_pairs:
        var1, var2 = pair
        factor_specs.append([var1, var2, check_var_id])
        factor_values.append(not_equal_tensor.flatten().tolist())
        check_var_id += 1
    last_check_var_id = check_var_id

    # and tensor
    and_tensor = torch.zeros(2, 2, 2)
    and_tensor[0,0,0] = 1.0 # true and true = true
    and_tensor[0,1,1] = 1.0 # true and false = false
    and_tensor[1,0,1] = 1.0 # false and true = false
    and_tensor[1,1,1] = 1.0 # false and false = false

    # comb factors # would need less iterations if the combination factors were built in a balanced tree way - but works for now
    check_var_id = num_cells # first check var id
    comb_var_id = last_check_var_id # first comb var id
    factor_specs.append([check_var_id, check_var_id+1, comb_var_id])
    factor_values.append(and_tensor.flatten().tolist())
    check_var_id += 2
    comb_var_id += 1
    for i in range(num_checks - 2): # k-1 additions for k summands
        factor_specs.append([check_var_id, comb_var_id -1, comb_var_id])
        factor_values.append(and_tensor.flatten().tolist())
        check_var_id += 1
        comb_var_id += 1
    output_factor_id = len(factor_specs) - 1  # index of output var factor

    # prior factors for the check vars
    first_prior_factor_id = len(factor_specs)
    for i in range(num_cells, num_cells + 2 * num_row_pairs + num_box_pairs):
        factor_specs.append([i])
        factor_values.append([0.5,0.5])
    last_prior_factor_id = len(factor_specs)

    ###############################################################################################
    ## build uai file
    lines = []
    lines.append(str(last_check_var_id) + ' ' + str(output_factor_id) + ' ' + str(first_prior_factor_id) + ' ' + str(last_prior_factor_id))
    lines.append(str(len(var_dims))) # num variables
    lines.append(' '.join(str(x) for x in var_dims)) # variable dimensions
    lines.append(str(len(factor_specs))) # num factors
    for spec in factor_specs:
        lines.append(str(len(spec)) + ' ' + ' '.join(str(x) for x in spec)) # size + var ids
    for values in factor_values:
        lines.append(str(len(values)) + ' ' + ' '.join(str(x) for x in values)) # factor values

    filename = f'problems/uai/Sudoku/cat_sudoku_{dimension}digit.uai'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))



if __name__ == "__main__":
    for dimension in [int(a) for a in sys.argv[1:]] or [2, 3, 4, 5, 6]:
        generate_uai_sudoku(dimension)
