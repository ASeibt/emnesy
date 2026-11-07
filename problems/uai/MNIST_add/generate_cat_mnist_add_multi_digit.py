""".uai factor graphs for MNIST addition. Run from the repo root with the numbers of digits
(default 1 to 100). First line of a file: the sum variable ids."""
import sys

import torch


def generate_uai_file(num_digits):
    # def variables
    # 0 - num_digits-1: left input digits
    # num_digits - 2*num_digits-1: right input digits
    # 2*num_digits - 3*num_digits-1: carry bits - the last carry is the most significant digit of the sum
    # 3*num_digits - 4*num_digits-1: sum digits
    var_dims = [10] * 2*num_digits + [2] * num_digits + [10] * num_digits

    # initialize lists for factor specs + values
    factor_specs = []
    factor_values = []

    ###############################################################################################
    ## def factor tensors

    # input factors
    for i in range(2*num_digits):
        factor_specs.append([i])
        factor_values.append([0.1] * 10) # uniform distribution

    # carry tensor
    carry_tensor = torch.zeros(10, 10, 2, 2)
    for d1 in range(10):
        for d2 in range(10):
            for old_carry in range(2):
                s = d1 + d2 + old_carry
                new_carry = 1 if s >= 10 else 0
                carry_tensor[d1, d2, old_carry, new_carry] = 1.0

    # carry tensor w/o old carry - for first carry factor
    carry_tensor_wo_old = torch.zeros(10, 10, 2)
    for d1 in range(10):
        for d2 in range(10):
            s = d1 + d2
            new_carry = 1 if s >= 10 else 0
            carry_tensor_wo_old[d1, d2, new_carry] = 1.0

    # sum_tensor
    sum_tensor = torch.zeros(10, 10, 2, 10)
    for d1 in range(10):
        for d2 in range(10):
            for c in range(2):
                s = d1 + d2 + c
                sum_digit = s % 10
                sum_tensor[d1, d2, c, sum_digit] = 1.0

    # sum tensor w/o carry - for first sum factor
    sum_tensor_wo_carry = torch.zeros(10, 10, 10)
    for d1 in range(10):  
        for d2 in range(10):
            s = d1 + d2
            sum_digit = s % 10
            sum_tensor_wo_carry[d1, d2, sum_digit] = 1.0

    ###############################################################################################
    ## def factor specs + values

    # first carry factor
    factor_specs.append([0, num_digits, 2*num_digits])
    factor_values.append(carry_tensor_wo_old.flatten().tolist())

    # first sum factor
    factor_specs.append([0, num_digits, 3*num_digits])
    factor_values.append(sum_tensor_wo_carry.flatten().tolist())

    # remaining carry + sum factors
    for i in range(1, num_digits):
        factor_specs.append([i, i + num_digits, i-1 + 2*num_digits, i + 2*num_digits])
        factor_values.append(carry_tensor.flatten().tolist())

        factor_specs.append([i, i + num_digits, i-1 + 2*num_digits, i + 3*num_digits])
        factor_values.append(sum_tensor.flatten().tolist())

    # sum prior factors - except msb sum digit
    for i in range(num_digits):
        factor_specs.append([3*num_digits + i])
        factor_values.append([0.1] * 10) # uniform distribution

    # prior factor of the most significant sum digit (= the last carry)
    factor_specs.append([3*num_digits - 1])
    factor_values.append([0.5, 0.5]) # uniform

    # output var_ids
    output_var_ids = list(range(3*num_digits, 4*num_digits))
    output_var_ids.append(3*num_digits - 1)

    ###############################################################################################
    ## build uai file
    lines = []
    lines.append(' '.join(str(x) for x in output_var_ids)) # output var ids
    lines.append(str(len(var_dims))) # num variables
    lines.append(' '.join(str(x) for x in var_dims)) # variable dimensions
    lines.append(str(len(factor_specs))) # num factors
    for spec in factor_specs:
        lines.append(str(len(spec)) + ' ' + ' '.join(str(x) for x in spec)) # size + var ids
    for values in factor_values:
        lines.append(str(len(values)) + ' ' + ' '.join(str(x) for x in values)) # factor values

    filename = f'problems/uai/MNIST_add/cat_mnist_add_{num_digits}digit.uai'
    with open(filename, 'w') as f:
        f.write('\n'.join(lines))



if __name__ == "__main__":
    for num_digits in [int(a) for a in sys.argv[1:]] or range(1, 101):
        generate_uai_file(num_digits)
