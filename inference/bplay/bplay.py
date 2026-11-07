"""BPLay: batched loopy belief propagation on a factor graph from a .uai file.

    bp = BPLay(batch_size, uai_file_type=3, data_path="....uai")
    bp.set_input_factor_ids(factor_ids, "forward")
    bp.set_output_var_ids(var_ids, "forward")
    bp.set_input(input_data, "forward")  # (input length, batch_size)
    for _ in range(num_iterations):
        bp.bplay_iteration()
    output = bp.get_output("forward")  # (output length, batch_size)

set_input resets the factors but not the messages (warm start from the previous run).
"""
import torch

from inference.bplay import bplay_from_uai_text_file, bplay_inputs_and_outputs, bplay_iteration

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BPLay:
    """damp: new = damp * old + (1 - damp) * update. clamp: min. factor value. amax: max-product.
    requires_grad: track gradients; note every call sets torch.set_grad_enabled(requires_grad) globally."""
    def __init__(self, batch_size, uai_file_type, damp=0, data_path=None, requires_grad=False, clamp=1e-20, amax=0, verbose=False):
        self.device = device
        self.batch_size = batch_size  # last dimension of all tensors
        self.damp = damp
        self.requires_grad = requires_grad
        self.clamp = clamp
        self.reduce = 'amax' if amax else 'sum'
        self.verbose = verbose

        # factor graph specification
        self.nbr_of_variables = -1
        self.nbr_of_factors = -1
        self.nbr_of_messages = -1

        # variable-factor-message links
        self.variable_edges = list()  # per variable: list of (factor_idx, message_idx)
        self.factor_edges = list()  # per factor: list of (variable_idx, message_idx)
        self.messages = list()  # per message: (variable_idx, factor_idx)

        # factor specifications
        self.factor_dims = None  # number of variables per factor
        self.var_dims = None  # number of states per variable
        self.msg_dims = None  # length of every message
        self.factor_shapes = None
        self.pos_in_factor_tensor = None
        self.factor_tensor = None  # flat; a factor of n variables appears n times (permuted)
        self.ORIGINAL_FACTOR_TENSOR = None  # never modified

        # message tensors
        self.ftv_messages = None  # factor-to-variable, flat
        self.vtf_messages = None  # variable-to-factor, flat
        self.pos_in_msg_tensor = None
        self.msg_norm_mask = None  # message id per entry

        # message passing masks
        self.ftv_msg_select_mask = None
        self.ftv_msg_mult_mask = None
        self.ftv_msg_marg_mask = None
        self.vtf_msg_select_mask = None
        self.vtf_msg_reduce_mask = None

        # input/output configurations, by name
        self.num_input_configs = 0
        self.num_output_configs = 0
        self.input_factor_ids = dict()
        self.input_factor_sizes = dict()
        self.input_length = dict()
        self.output_variables = dict()
        self.output_length = dict()

        # input/output masks, by name
        self.input_to_factor_tensor_gather_mask = dict()
        self.input_to_factor_tensor_scatter_mask = dict()
        self.ftv_msg_to_output_gather_mask = dict()
        self.ftv_msg_to_output_scatter_mask = dict()
        self.output_norm_mask = dict()

        if data_path:
            self.load_from_uai_text_file(data_path, uai_file_type)

    load_from_uai_text_file = bplay_from_uai_text_file.bplay_from_uai_textfile

    set_input_factor_ids = bplay_inputs_and_outputs.set_input_factor_ids
    set_output_var_ids = bplay_inputs_and_outputs.set_output_var_ids
    set_input = bplay_inputs_and_outputs.set_input
    get_output = bplay_inputs_and_outputs.get_output

    flat_normalize = bplay_iteration.flat_normalize
    bplay_iteration_w_checkpointing = bplay_iteration.bplay_iteration_w_checkpointing
    bplay_iteration = bplay_iteration.bplay_iteration
