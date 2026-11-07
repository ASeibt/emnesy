"""Reads the factor graph of a BPLay instance from a .uai text file and builds its tensors and masks."""

import torch
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# .uai file types, differing in the notation of the factor values:
# 1: size / values / empty line (e.g. DBN, Grids)
# 2: size / values (e.g. Alchemy, Promedus)
# 3: size and values in one line, no empty line after the factor specs (used here)
# 4: size / values over several indented lines / empty line (e.g. blockmap, linkage)

def bplay_from_uai_textfile(self, data_path, uai_file_type):

    torch.set_grad_enabled(self.requires_grad)
    log = print if self.verbose else (lambda *args, **kwargs: None)

    # read the file to a list
    with open(data_path, 'r', encoding='utf-8') as file:
        next(file)
        lines = file.readlines()
    log(f"reading {data_path}...")
        
    # number of variables
    string_values = lines[0].strip().split()
    del lines[0]
    if len(string_values) != 1:
        raise ValueError(f"Expected exactly one value in the second line of {data_path}, found {len(string_values)}")
    else:
        self.nbr_of_variables = int(string_values[0])
    self.variable_edges = [[] for _ in range(self.nbr_of_variables)]

    # extract variable dims
    self.var_dims = list(map(int, lines[0].strip().split()))
    del lines[0]
    if len(self.var_dims) != self.nbr_of_variables:
        raise ValueError(f"Expected {self.nbr_of_variables} values in the third line of {data_path}, found {len(self.var_dims)}")
    log(self.nbr_of_variables, "variables of dims ", min(self.var_dims), "to",  max(self.var_dims))
        
    # number of factors
    string_values = lines[0].strip().split()
    del lines[0]
    if len(string_values) != 1:
        raise ValueError(f"Expected exactly one value in the fourth line of {data_path}, found {len(string_values)}")
    else:
        self.nbr_of_factors = int(string_values[0])
    self.factor_edges = [[] for _ in range(self.nbr_of_factors)]
    self.factor_dims = [[] for _ in range(self.nbr_of_factors)]
    self.factor_shapes = [[] for _ in range(self.nbr_of_factors)]

    # extract factor specifications
    string_factor_specs = lines[:self.nbr_of_factors]
    del lines[:self.nbr_of_factors]
    factor_specs = [list(map(int, string_spec.strip().split())) for string_spec in string_factor_specs]
    if len(factor_specs) != self.nbr_of_factors:
        raise ValueError(f"Expected {self.nbr_of_factors} factor specifications in the next {self.nbr_of_factors} lines of {data_path}, found {len(factor_specs)}")
        
    # extract factor values
    if uai_file_type != 3:
        del lines[0]

    if uai_file_type == 1:
        string_factor_values = [lines[i + 1].strip().split() for i in range(0, len(lines)-1, 3)]
    elif uai_file_type == 2:
        string_factor_values = [lines[i + 1].strip().split() for i in range(0, len(lines)-1, 2)]
    elif uai_file_type == 3:
        string_factor_values = [line.strip().split()[1:] for line in lines]
    elif uai_file_type == 4:
        string_factor_values = []

        i = 0 # current line
        
        while i < len(lines):
            current_block = []
            i += 1

            while i < len(lines): # collect values until...
                line = lines[i]
                if line.strip() == "": # ... there is an empty line
                    i += 1 # skip empty line
                    break
                current_block.extend(line.strip().split())
                i += 1
            string_factor_values.append(current_block)
    else:
        raise ValueError(f"uai_file_type has to be in {1,2,3,4}, found {uai_file_type}")
    
    factor_values = [list(map(float, string_factor_value)) for string_factor_value in string_factor_values]
    if len(factor_values) != self.nbr_of_factors:
        raise ValueError(f"Expected {self.nbr_of_factors} factor values in the next lines of {data_path}, found {len(factor_values)}")

###################################################################################################
    ## from here on it's no longer uai dataset specific
       
    message_idx = 0
    pos_in_msg_tensor = list()
    self.msg_dims = list()
    msg_norm_list = list()
    m_pos = 0

    for factor_idx, (factor_spec, factor_value) in enumerate(zip(factor_specs, factor_values)):

        # extract factor dims
        self.factor_dims[factor_idx] = factor_spec[0] # nbr of var's in the factor

        # factor shapes, variable-factor-message links, message positions
        for var_idx_in_factor in range(int(self.factor_dims[factor_idx])):
            var_idx = factor_spec[var_idx_in_factor+1]

            self.factor_shapes[factor_idx].append(self.var_dims[var_idx])

            self.variable_edges[var_idx].append((factor_idx, message_idx))
            self.factor_edges[factor_idx].append((var_idx, message_idx))
            self.messages.append((var_idx, factor_idx))
            self.msg_dims.append(self.var_dims[var_idx])
            pos_in_msg_tensor.append(m_pos)
            msg_norm_list.extend([message_idx] * self.var_dims[var_idx]) # extend the msg_norm_list

            m_pos += self.var_dims[var_idx]
            message_idx += 1

    if len(self.factor_dims) != self.nbr_of_factors:
        raise ValueError(f"Expected {self.nbr_of_factors} factor dims, found {len(self.factor_dims)}")
    if len(self.factor_shapes) != self.nbr_of_factors:
        raise ValueError(f"Expected {self.nbr_of_factors} factor shapes, found {len(self.factor_shapes)}")
    log(self.nbr_of_factors, "factors of dims ", min(self.factor_dims), "to", max(self.factor_dims))

    # extract nbr of messages
    self.nbr_of_messages = message_idx
    log(self.nbr_of_messages, "messages of total length", m_pos)

    # initialize lists for factor tensor and masks
    factor_tensor_list = list()
    ftv_msg_select_mask_list = list()
    ftv_msg_mult_mask_list = list()
    ftv_msg_marg_mask_list = list()
    vtf_msg_select_mask_list = list()   # list of source msg ids
    vtf_msg_reduce_mask_list = list()   # list of target msg ids

    pos_in_factor_tensor = list()
    f_pos = 0

###################################################################################################
    ## fill factor tensor and mask lists
    for factor_idx, (factor_spec, factor_value) in enumerate(zip(factor_specs, factor_values)):

        if factor_idx % 1000 == self.nbr_of_factors % 1000 - 1:
            log("read", factor_idx+1, "of", self.nbr_of_factors, "factors", end="\r")
            
        # permute on cpu, send to device later
        factor_value_tensor = torch.tensor(factor_value, device=torch.device("cpu"), dtype=torch.float).view(*self.factor_shapes[factor_idx])
        dimensions = list(range(self.factor_dims[factor_idx])) # 0,1,...,factor_dim-1

        # add factor position in factor tensor
        pos_in_factor_tensor.append(f_pos)

        for var_idx_in_factor in range(self.factor_dims[factor_idx]):
            
            # fill factor_tensor_list
            rotated_dims = dimensions[var_idx_in_factor:] + dimensions[:var_idx_in_factor]
            permuted_factor_value_tensor = factor_value_tensor.permute(rotated_dims)
            factor_tensor_list.append(permuted_factor_value_tensor.flatten())

            # fill ftv_msg_marg_mask_list
            var_idx = factor_spec[rotated_dims[0]+1]
            msg_idx = next(m_idx for f_idx, m_idx in self.variable_edges[var_idx] if f_idx == factor_idx)
            m_pos = pos_in_msg_tensor[msg_idx]
            target_pos = torch.arange(m_pos, m_pos + self.var_dims[var_idx], device=torch.device("cpu"), dtype=torch.int)
            marg_tensor = target_pos.view(-1, *[1] * (self.factor_dims[factor_idx]-1)).expand_as(permuted_factor_value_tensor).flatten()
            ftv_msg_marg_mask_list.append(marg_tensor)

            # fill ftv msg select+mult mask lists
            for mult_var_idx_in_factor in range(1, self.factor_dims[factor_idx]):
                mult_var_idx = factor_spec[1+rotated_dims[mult_var_idx_in_factor]]
                mult_msg_idx = next(msg_idx for f_idx, msg_idx in self.variable_edges[mult_var_idx] if f_idx == factor_idx)
                mult_m_pos = pos_in_msg_tensor[mult_msg_idx]

                # select mask - where it comes from
                source_pos = torch.arange(mult_m_pos, mult_m_pos + self.var_dims[mult_var_idx], device=torch.device("cpu"), dtype=torch.int)
                source_tensor = source_pos.view(*[1] * mult_var_idx_in_factor, -1, *[1] * (self.factor_dims[factor_idx]-mult_var_idx_in_factor-1)).expand_as(permuted_factor_value_tensor).flatten()
                ftv_msg_select_mask_list.append(source_tensor)

                # mult mask - where it goes to
                target_tensor = torch.arange(f_pos, f_pos + math.prod(self.factor_shapes[factor_idx]), device=torch.device("cpu"), dtype=torch.int).flatten()
                ftv_msg_mult_mask_list.append(target_tensor)

            # fill msg_select/reduce_mask_stack_lists
            for other_factor_of_this_var, m_idx in self.variable_edges[var_idx]:
                if not msg_idx == m_idx:
                    other_m_positions = torch.arange(pos_in_msg_tensor[m_idx], pos_in_msg_tensor[m_idx] + self.var_dims[var_idx], device=torch.device("cpu"), dtype=int)
                    for other_m_pos, m_pos in zip(other_m_positions, target_pos):
                        vtf_msg_select_mask_list.append(other_m_pos.item()) # where it comes from
                        vtf_msg_reduce_mask_list.append(m_pos.item()) # where it goes to

            f_pos += math.prod(self.factor_shapes[factor_idx])

###################################################################################################
    ## build all tensors and send to device

    # turn lists to tensors + send to device
    self.factor_dims = torch.tensor(self.factor_dims, device=device, dtype=torch.int)
    self.var_dims = torch.tensor(self.var_dims, device=device, dtype=torch.int)
    self.msg_dims = torch.tensor(self.msg_dims, device=device, dtype=torch.int)
    self.pos_in_factor_tensor = torch.tensor(pos_in_factor_tensor, device=device, dtype=torch.int)
    self.pos_in_msg_tensor = torch.tensor(pos_in_msg_tensor, device=device, dtype=torch.int)
    self.msg_norm_mask = torch.tensor(msg_norm_list, device=device, dtype=torch.int)

    # cat + expand factor_tensor
    self.factor_tensor = torch.cat(factor_tensor_list).unsqueeze(-1).expand(-1, self.batch_size).clone().to(device)
    self.ORIGINAL_FACTOR_TENSOR = self.factor_tensor.clone() # will never be modified
    log("factor tensor shape:", self.factor_tensor.shape)

    # cat ftv_msg_select/mult/marg_mask_lists to tensors
    self.ftv_msg_select_mask = torch.cat(ftv_msg_select_mask_list).to(device)
    self.ftv_msg_mult_mask = torch.cat(ftv_msg_mult_mask_list).to(device)
    self.ftv_msg_marg_mask = torch.cat(ftv_msg_marg_mask_list).to(device) # shape same as factor_tensor
    log("ftv msg select/mult/marg masks shape:", self.ftv_msg_select_mask.shape, self.ftv_msg_mult_mask.shape, self.ftv_msg_marg_mask.shape)

    # create vtf_msg_select/reduce_masks 
    self.vtf_msg_select_mask = torch.tensor(vtf_msg_select_mask_list, device=device, dtype=torch.int)
    self.vtf_msg_reduce_mask = torch.tensor(vtf_msg_reduce_mask_list, device=device, dtype=torch.int)
    log("vtf msg select/reduce masks shape:", self.vtf_msg_select_mask.shape, self.vtf_msg_reduce_mask.shape)

    # initialize message tensors
    self.vtf_messages = torch.ones(self.msg_dims.sum(), self.batch_size, device=device)
    self.ftv_messages = torch.ones(self.msg_dims.sum(), self.batch_size, device=device)

    log("reading uai text file done.")
