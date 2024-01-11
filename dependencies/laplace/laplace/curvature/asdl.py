from abc import abstractproperty
import warnings
from asdfghjkl.kernel import linear_network_kernel, linear_network_kernel_indep, empirical_network_kernel
import numpy as np
import torch

from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG
from asdfghjkl.fisher import fisher, zero_fisher
from asdfghjkl.gradient import batch_gradient

from laplace.curvature import CurvatureInterface, GGNInterface, EFInterface
from laplace.matrix import Kron
from laplace.utils import _is_batchnorm


class AsdlInterface(CurvatureInterface):
    """Interface for asdfghjkl backend.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True, kron_jac=False):
        self.kron_jac = kron_jac
        super().__init__(model, likelihood, last_layer, differentiable)

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        Js = list()
        for i in range(self.model.output_size):
            def loss_fn(outputs, targets):
                return outputs[:, i].sum()

            f = batch_gradient(self.model, loss_fn, x, None, self.kron_jac, **self.backward_kwargs)
            Js.append(_get_batch_grad(self.model, self.kron_jac))
        Js = torch.stack(Js, dim=1)

        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def single_jacobians(self, x, output_ix):
        def loss_fn(outputs, targets):
            if output_ix.ndim == 0:  # scalar
                return outputs[:, output_ix].sum()
            elif output_ix.ndim == 1:  # vector iid
                return outputs.gather(1, output_ix.unsqueeze(-1)).sum()
            else:
                raise ValueError('output_ix must be scalar or vector')

        f = batch_gradient(self.model, loss_fn, x, None, self.kron_jac, **self.backward_kwargs)
        Js = _get_batch_grad(self.model, self.kron_jac)
        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def mean_jacobians(self, x):
        assert self.likelihood == 'heteroscedastic_regression'
        def loss_fn(outputs, targets):
            m = - outputs[:, 0] / (2 * outputs[:, 1])
            return m.sum()
        f = batch_gradient(self.model, loss_fn, x, None, self.kron_jac, **self.backward_kwargs)
        Js = _get_batch_grad(self.model, self.kron_jac)
        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter
        \\(\\theta\\) using asdfghjkl's backend.

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        y : torch.Tensor

        Returns
        -------
        loss : torch.Tensor
        Gs : torch.Tensor
            gradients `(batch, parameters)`
        """
        f = batch_gradient(self.model, self.lossfunc, x, y, self.kron_jac, **self.backward_kwargs)
        Gs = _get_batch_grad(self._model, self.kron_jac)
        loss = self.lossfunc(f, y)
        if self.differentiable:
            return Gs, loss
        return Gs.detach(), loss.detach()

    @abstractproperty
    def _ggn_type(self):
        raise NotImplementedError()

    def _get_kron_factors(self, curv, M):
        kfacs = list()
        for module in curv._model.modules():
            if _is_batchnorm(module):
                warnings.warn('BatchNorm unsupported for Kron, ignore.')
                continue

            stats = getattr(module, self._ggn_type, None)
            if stats is None:
                continue
            if hasattr(module, 'bias') and module.bias is not None:
                # split up bias and weights
                kfacs.append([stats.kron.B, stats.kron.A[:-1, :-1]])
                kfacs.append([stats.kron.B * stats.kron.A[-1:, -1:].clone() / M])
            elif hasattr(module, 'weight'):
                p, q = np.prod(stats.kron.B.shape), np.prod(stats.kron.A.shape)
                if p == q == 1:
                    kfacs.append([stats.kron.B * stats.kron.A])
                else:
                    kfacs.append([stats.kron.B, stats.kron.A])
            else:
                raise ValueError(f'Whats happening with {module}?')
        return Kron(kfacs)

    @staticmethod
    def _rescale_kron_factors(kron, N):
        for F in kron.kfacs:
            if len(F) == 2:
                F[1] *= 1/N
        return kron

    def diag(self, X, y, **kwargs):
        if self.last_layer:
            f, X = self.model.forward_with_features(X)
        f, curv = fisher(self._model, self._ggn_type, SHAPE_DIAG, likelihood=self.likelihood,
                         inputs=X, targets=y, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        diag_ggn = curv.matrices_to_vector(None)

        if self.differentiable:
            return self.factor * loss, self.factor * diag_ggn
        return self.factor * loss.detach(), self.factor * diag_ggn.detach()

    def kron(self, X, y, N, **kwargs):
        if self.last_layer:
            f, X = self.model.forward_with_features(X)
        f, curv = fisher(self._model, self._ggn_type, SHAPE_KRON, likelihood=self.likelihood,
                         inputs=X, targets=y, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)
        zero_fisher(self._model, [self._ggn_type])

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron.detach()


class AsdlGGN(AsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` using asdfghjkl.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True, kron_jac=False, stochastic=False):
        super().__init__(model, likelihood, last_layer, differentiable, kron_jac)
        self.stochastic = stochastic

    @property
    def _ggn_type(self):
        return FISHER_MC if self.stochastic else FISHER_EXACT

    def kernel(self, x, y, prec_diag, **kwargs):
        if self.last_layer:
            raise ValueError('Unsupported last layer for kernel')

        f, K = linear_network_kernel(self._model, x, scale=1 / prec_diag, likelihood=self.likelihood,
                                     differentiable=self.differentiable, kron_jac=self.kron_jac)
        n, c = f.shape
        K = K.transpose(1, 2).reshape(n*c, n*c)  # n x c x n x c

        loss = self.factor * self.lossfunc(f, y)
        if self.differentiable:
            return loss, K
        return loss.detach(), K.detach()

    def indep_kernel(self, x, y, prec_diag):
        if self.last_layer:
            raise ValueError('Unsupported last layer for kernel')
        f, K = linear_network_kernel_indep(self._model, x, scale=1 / prec_diag, likelihood=self.likelihood,
                                           differentiable=self.differentiable, kron_jac=self.kron_jac)
        loss = self.factor * self.lossfunc(f, y)
        if self.differentiable:
            return loss, K
        return loss.detach(), K.detach()

    def single_kernel(self, x, y, prec_diag, output_ix):
        if self.last_layer:
            raise ValueError('Unsupported last layer for kernel')
        f, K = linear_network_kernel_indep(
            self._model, x, scale=1 / prec_diag, likelihood=self.likelihood, differentiable=self.differentiable,
            kron_jac=self.kron_jac, single_output=output_ix
        )
        loss = self.factor * self.lossfunc(f, y)
        if self.differentiable:
            return loss, K
        return loss.detach(), K.detach()

    def single_kron(self, X, y, N, output_ix, **kwargs):
        if self.last_layer:
            f, X = self.model.forward_with_features(X)
        f, curv = fisher(self._model, self._ggn_type, SHAPE_KRON, likelihood=self.likelihood,
                         inputs=X, targets=y, single_output=output_ix, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)
        zero_fisher(self._model, [self._ggn_type])

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron.detach()

    def single_diag(self, X, y, output_ix, **kwargs):
        if self.last_layer:
            f, X = self.model.forward_with_features(X)
        f, curv = fisher(self._model, self._ggn_type, SHAPE_DIAG, likelihood=self.likelihood,
                         inputs=X, targets=y, single_output=output_ix, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        diag_ggn = curv.matrices_to_vector(None)

        if self.differentiable:
            return self.factor * loss, self.factor * diag_ggn
        return self.factor * loss.detach(), self.factor * diag_ggn.detach()


class AsdlEF(AsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using asdfghjkl.
    """

    @property
    def _ggn_type(self):
        return COV

    def kernel(self, x, y, prec_diag, prec, prec_structure):
        if self.last_layer:
            raise ValueError('Unsupported last layer for kernel')

        if prec_structure == 'diagonal':
            return super().kernel(x, y, prec_diag)

        loss, K = empirical_network_kernel(self._model, x, y, self.lossfunc, 1 / prec,
                                           differentiable=self.differentiable, kron_jac=self.kron_jac)
        if self.differentiable:
            return self.factor * loss, self.factor * K
        return self.factor * loss.detach(), self.factor * K.detach()


def _flatten_after_batch(tensor: torch.Tensor):
    if tensor.ndim == 1:
        return tensor.unsqueeze(-1)
    else:
        return tensor.flatten(start_dim=1)


def _get_batch_grad(model, kron_jac=False):
    field = 'batch_grads_kron' if kron_jac else 'batch_grads'
    batch_grads = list()
    for module in model.modules():
        if hasattr(module, 'op_results'):
            res = module.op_results[field]
            if 'weight' in res:
                batch_grads.append(_flatten_after_batch(res['weight']))
            if 'bias' in res:
                batch_grads.append(_flatten_after_batch(res['bias']))
            if len(set(res.keys()) - {'weight', 'bias'}) > 0:
                raise ValueError(f'Invalid parameter keys {res.keys()}')
    return torch.cat(batch_grads, dim=1)
