import torch
from torch import nn

from .operation import Operation


class Bias(nn.Module):
    def __init__(self):
        super(Bias, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 0)

    def forward(self, input):
        return input + self.weight


class BiasExt(Operation):
    """
    module.fixup_bias: 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out
    """
    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return out_grads.view(N, -1).sum(dim=1)

    @staticmethod
    def batch_grads_kron_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return out_grads.view(N, -1).sum(dim=1)

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return out_grads.view(N, -1).sum(dim=1).square().sum()

    @staticmethod
    def cov_kron_A(module, in_data):
        return torch.ones((1, 1), device=in_data.device)

    @staticmethod
    def cov_kron_B(module, out_grads):
        N = out_grads.size(0)
        grad_grad = out_grads.view(N, -1).sum(dim=1).square().sum()
        return grad_grad.unsqueeze((0))

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        N1, N2 = in_data1.size(0), in_data2.size(0)
        return in_data1.new_ones(N1, N2)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        N1, N2 = out_grads1.size(0), out_grads2.size(0)
        grad1 = out_grads1.view(N1, -1).sum(dim=1)
        grad2 = out_grads2.view(N2, -1).sum(dim=1)
        return torch.outer(grad1, grad2)  # n x n
