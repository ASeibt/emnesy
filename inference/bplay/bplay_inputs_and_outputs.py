"""Named input/output configurations of BPLay: input factors and output variables."""

import torch
import math

###################################################################################################
## def input/output

def set_input_factor_ids(self, factor_ids, name):

    torch.set_grad_enabled(self.requires_grad)

    self.num_input_configs += 1
    self.input_factor_ids[name] = factor_ids

    # compute input sizes and length
    input_factor_sizes =[math.prod(self.factor_shapes[factor_idx]) for factor_idx in factor_ids]
    self.input_factor_sizes[name] = torch.tensor(input_factor_sizes, device=self.device, dtype=torch.int)
    self.input_length[name] = torch.sum(self.input_factor_sizes[name])
    if self.verbose:
        print(name, "input length:", self.input_length[name].item())

    # input -> factor tensor: gather mask (with permutation)
    pos_in_input = 0
    input_gather_mask_list = list()

    for idx_in_input, factor_idx in enumerate(self.input_factor_ids[name]):
        input_segment = torch.arange(pos_in_input, pos_in_input + self.input_factor_sizes[name][idx_in_input], device=self.device, dtype=torch.int64).reshape(self.factor_shapes[factor_idx]) # reshape to factor shape for permuting
        pos_in_input += self.input_factor_sizes[name][idx_in_input]
        dimensions = torch.arange(self.factor_dims[factor_idx].item(), device=self.device, dtype=torch.int)

        for var_idx_in_factor in range(self.factor_dims[factor_idx].item()): # every factor appears once per variable
            rotated_dimensions = torch.roll(dimensions, shifts=var_idx_in_factor)
            rotated_segment = input_segment.permute(rotated_dimensions.tolist()).flatten()
            input_gather_mask_list.append(rotated_segment)

    self.input_to_factor_tensor_gather_mask[name] = torch.cat(input_gather_mask_list)

    # input -> factor tensor: scatter mask
    input_scatter_mask_list = list()

    for idx_in_input, factor_idx in enumerate(self.input_factor_ids[name]):
        factor_size = self.input_factor_sizes[name][idx_in_input].item()
        factor_dim = self.factor_dims[factor_idx].item()
        pos = self.pos_in_factor_tensor[factor_idx]
        
        factor_tensor_ids = torch.arange(pos, pos + factor_dim * factor_size, device=self.device, dtype=torch.int64)
        input_scatter_mask_list.append(factor_tensor_ids)

    self.input_to_factor_tensor_scatter_mask[name] = torch.cat(input_scatter_mask_list)


def set_output_var_ids(self, var_ids, name):

    torch.set_grad_enabled(self.requires_grad)

    # set output variables
    self.num_output_configs += 1
    self.output_variables[name] = var_ids

    self.output_length[name] = self.var_dims[var_ids].sum()

    # masks for the output marginals and their normalization
    output_gather_mask_list = list()
    output_scatter_mask_list = list()
    output_norm_mask_list = list()
    pos_in_output = 0

    for var_idx_in_output, var_idx in enumerate(var_ids):
        var_dim = self.var_dims[var_idx].item()
        output_norm_mask_list.append(var_idx_in_output * torch.ones(var_dim, device=torch.device("cpu"), dtype=torch.int64))
        for f_idx, m_idx in self.variable_edges[var_idx]:
            m_pos = self.pos_in_msg_tensor[m_idx].item()
            output_gather_mask_list.append(torch.arange(m_pos, m_pos + var_dim, device=torch.device("cpu"), dtype=torch.int64))
            output_scatter_mask_list.append(torch.arange(pos_in_output, pos_in_output + var_dim, device=torch.device("cpu"), dtype=torch.int64))
        pos_in_output += var_dim

    self.ftv_msg_to_output_gather_mask[name] = torch.cat(output_gather_mask_list).to(self.device)
    self.ftv_msg_to_output_scatter_mask[name] = torch.cat(output_scatter_mask_list).to(self.device)
    self.output_norm_mask[name] = torch.cat(output_norm_mask_list).to(self.device)

###################################################################################################
## set/get input/output

def set_input(self, input_data, name):  # factor values in the order of set_input_factor_ids
    torch.set_grad_enabled(self.requires_grad)
    
    # shape: (sum of the input factor sizes, batch_size)
    if input_data.shape != (self.input_length[name], self.batch_size):
        raise ValueError(f"input_data is expected to be of shape (self.input_length ({self.input_length[name]}), self.batch_size ({self.batch_size})) but found shape {input_data.shape}")

    # gather-scatter the input into the factor tensor
    to_scatter = torch.gather(input_data,
                              dim=0,
                              index=self.input_to_factor_tensor_gather_mask[name].unsqueeze(-1).expand(-1, self.batch_size)) # gather
    self.factor_tensor = self.ORIGINAL_FACTOR_TENSOR.clone()  # reset the factors, not the messages
    self.factor_tensor.scatter_(dim=0,
                                index=self.input_to_factor_tensor_scatter_mask[name].unsqueeze(-1).expand(-1, self.batch_size),
                                src=to_scatter)


def get_output(self, name):
    torch.set_grad_enabled(self.requires_grad)
    
    to_scatter = torch.gather(self.ftv_messages,
                              dim=0,
                              index=self.ftv_msg_to_output_gather_mask[name].unsqueeze(-1).expand(-1, self.batch_size)) # gather
    output = torch.scatter_reduce(torch.ones(self.output_length[name], self.batch_size, device=self.device),
                                  0,
                                  self.ftv_msg_to_output_scatter_mask[name].unsqueeze(-1).expand(-1, self.batch_size),
                                  to_scatter,
                                  reduce="prod",
                                  include_self=False)
    return self.flat_normalize(output, self.output_norm_mask[name], len(self.output_variables[name]))
