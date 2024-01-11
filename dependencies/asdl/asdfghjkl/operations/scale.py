import torch
from torch import nn

from .operation import Operation


class Scale(nn.Module):
    def __init__(self):
        super(Scale, self).__init__()
        self.weight = nn.Parameter(torch.ones(1))
        
    def reset_parameters(self):
        nn.init.constant_(self.weight, 1)
        
    def forward(self, input):
        return self.weight * input


class ScaleExt(Operation):
    """
    module.weight: 1

    Argument shapes
    in_data: n x f_in
    out_grads: n x f_out = f_in
    """
    @staticmethod
    def batch_grads_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return (out_grads * in_data).view(N, -1).sum(dim=1)

    @staticmethod
    def batch_grads_kron_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return (out_grads * in_data).view(N, -1).sum(dim=1)

    @staticmethod
    def cov_diag_weight(module, in_data, out_grads):
        N = out_grads.size(0)
        return (out_grads * in_data).view(N, -1).sum(dim=1).square().sum()

    @staticmethod
    def cov_kron_A(module, in_data):
        setattr(module, 'n_in_data', in_data)
        return torch.ones(1, 1, device=in_data.device) 

    @staticmethod
    def cov_kron_B(module, out_grads):
        N = out_grads.size(0)
        return (module.n_in_data * out_grads).view(N, -1).sum(dim=1).square().sum().view(1, 1)

    @staticmethod
    def gram_A(module, in_data1, in_data2):
        setattr(module, 'n_in_data1', in_data1)
        setattr(module, 'n_in_data2', in_data2)
        N1, N2 = in_data1.size(0), in_data2.size(0)
        return in_data1.new_ones(N1, N2)  # n x n

    @staticmethod
    def gram_B(module, out_grads1, out_grads2):
        N1, N2 = out_grads1.size(0), out_grads2.size(0)
        grad1 = (module.n_in_data1 * out_grads1).view(N1, -1).sum(dim=1)
        grad2 = (module.n_in_data2 * out_grads2).view(N2, -1).sum(dim=1)
        return torch.outer(grad1, grad2)
