"""One BP iteration: update all factor-to-variable (ftv), then all variable-to-factor (vtf) messages.
Products are sums in log space; messages are damped and normalized."""
import torch
import torch.utils.checkpoint as cp


def bplay_iteration(self):
    """Without gradients (E-step, evaluation)."""
    torch.set_grad_enabled(self.requires_grad)
    self.ftv_messages = update_ftv_messages(self.vtf_messages, self.ftv_messages, self.factor_tensor, self.ftv_msg_select_mask,
                                            self.ftv_msg_mult_mask, self.ftv_msg_marg_mask, self.batch_size, self.damp, self.device,
                                            self.msg_norm_mask, self.nbr_of_messages, clamp=self.clamp, msg_clamp=1e-20, reduce=self.reduce)
    self.vtf_messages = update_vtf_messages(self.ftv_messages, self.vtf_messages, self.vtf_msg_select_mask, self.vtf_msg_reduce_mask,
                                            self.damp, self.device, self.msg_norm_mask, self.nbr_of_messages)


def bplay_iteration_w_checkpointing(self):
    """With gradient checkpointing (end-to-end training). Clamps vtf messages at self.clamp, not 1e-20."""
    torch.set_grad_enabled(self.requires_grad)
    self.ftv_messages = cp.checkpoint(update_ftv_messages, self.vtf_messages, self.ftv_messages, self.factor_tensor, self.ftv_msg_select_mask,
                                      self.ftv_msg_mult_mask, self.ftv_msg_marg_mask, self.batch_size, self.damp, self.device,
                                      self.msg_norm_mask, self.nbr_of_messages, clamp=self.clamp, msg_clamp=self.clamp, reduce=self.reduce,
                                      use_reentrant=False)
    self.vtf_messages = cp.checkpoint(update_vtf_messages, self.ftv_messages, self.vtf_messages, self.vtf_msg_select_mask,
                                      self.vtf_msg_reduce_mask, self.damp, self.device, self.msg_norm_mask, self.nbr_of_messages,
                                      use_reentrant=False)


def flat_normalize(self, msg_tensor, mask, nbr_of_msgs):
    return flat_normalize_w_args(msg_tensor, mask, nbr_of_msgs)


def update_ftv_messages(vtf_messages, ftv_messages, factor_tensor, ftv_msg_select_mask, ftv_msg_mult_mask, ftv_msg_marg_mask,
                        batch_size, damp, device, msg_norm_mask, nbr_of_messages, clamp, msg_clamp, reduce):
    """Multiply factors with incoming messages, then sum (or max) out the other variables.
    clamp / msg_clamp: min. factor / message value before the log."""
    to_multiply = torch.index_select(vtf_messages, 0, ftv_msg_select_mask)
    prod_tensor = factor_tensor.clamp(min=clamp).log().scatter_reduce(0,
                                                                      ftv_msg_mult_mask.long().unsqueeze(-1).expand(-1, batch_size),
                                                                      to_multiply.clamp(min=msg_clamp).log(),
                                                                      reduce='sum',
                                                                      include_self=True).exp()
    update = torch.scatter_reduce(torch.ones_like(ftv_messages, device=device),
                                  0,
                                  ftv_msg_marg_mask.long().unsqueeze(-1).expand(-1, batch_size),
                                  prod_tensor,
                                  reduce=reduce,
                                  include_self=False).reshape(ftv_messages.shape)
    ftv_messages = damp * ftv_messages + (1 - damp) * update
    return flat_normalize_w_args(ftv_messages, msg_norm_mask, nbr_of_messages)


def update_vtf_messages(ftv_messages, vtf_messages, vtf_msg_select_mask, vtf_msg_reduce_mask, damp, device, msg_norm_mask, nbr_of_messages):
    """Product of the messages from the other factors."""
    to_multiply = torch.index_select(ftv_messages, 0, vtf_msg_select_mask)
    update = torch.scatter_reduce(torch.ones_like(vtf_messages, device=device),
                                  0,
                                  vtf_msg_reduce_mask.long().unsqueeze(-1).expand_as(to_multiply),
                                  to_multiply.clamp(min=1e-20).log(),
                                  'sum',
                                  include_self=False).exp()
    vtf_messages = damp * vtf_messages + (1 - damp) * update
    return flat_normalize_w_args(vtf_messages, msg_norm_mask, nbr_of_messages)


def flat_normalize_w_args(msg_tensor, mask, nbr_of_msgs):
    """Normalize every message to sum to one."""
    norm_constants = torch.scatter_reduce(torch.ones(nbr_of_msgs, msg_tensor.shape[1], device=msg_tensor.device),
                                          0,
                                          mask.long().unsqueeze(-1).expand(-1, msg_tensor.shape[1]),
                                          msg_tensor,
                                          reduce='sum',
                                          include_self=False)
    return msg_tensor / torch.index_select(norm_constants, 0, mask).clamp(min=1e-20)
