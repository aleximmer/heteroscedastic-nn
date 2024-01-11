from math import log, pi
import torch

C = - 0.5 * log(2 * pi)


def heteroscedastic_mse_loss(input, target, reduction='mean'):
    """Heteroscedastic negative log likelihood Normal.

    Parameters
    ----------
    input : torch.Tensor (n, 2)
        two natural parameters per data point
    target : torch.Tensor (n, 1)
        targets
    """
    assert input.ndim == target.ndim == 2
    assert input.shape[0] == target.shape[0]
    n, _ = input.shape
    target = torch.cat([target, target.square()], dim=1)
    inner = torch.einsum('nk,nk->n', target, input)
    log_A = input[:, 0].square() / (4 * input[:, 1]) + 0.5 * torch.log(- 2 * input[:, 1])
    log_lik = n * C + inner.sum() + log_A.sum()
    if reduction == 'mean':
        return - log_lik / n
    elif reduction == 'sum':
        return - log_lik 
    else:
        raise ValueError('Invalid reduction', reduction)
