import warnings
import numpy as np
import torch

from asdfghjkl.gradient import batch_aug_gradient
from asdfghjkl.fisher import fisher
from asdfghjkl.kernel import linear_network_kernel, linear_network_kernel_indep, empirical_network_kernel
from asdfghjkl import FISHER_EXACT, FISHER_MC, COV
from asdfghjkl import SHAPE_KRON, SHAPE_DIAG

from laplace.curvature import CurvatureInterface, GGNInterface
from laplace.curvature.asdl import _get_batch_grad
from laplace.curvature import EFInterface
from laplace.matrix import Kron
from laplace.utils import _is_batchnorm


class AugAsdlInterface(CurvatureInterface):
    """Interface for Backpack backend when using augmented Laplace.
    This ensures that Jacobians, gradients, and the Hessian approximation remain differentiable
    and deals with S-augmented sized inputs (additional to the batch-dimension).
    """

    def __init__(self, model, likelihood, last_layer=False, differentiable=True, kron_jac=False):
        self.kron_jac = kron_jac
        super().__init__(model, likelihood, last_layer, differentiable)

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using asdfghjkl's gradient per output dimension, averages over aug dimension.

        Parameters
        ----------
        model : torch.nn.Module
        x : torch.Tensor
            input data `(batch, n_augs, input_shape)` on compatible device with model.

        Returns
        -------
        Js : torch.Tensor
            averaged Jacobians over `n_augs` of shape `(batch, parameters, outputs)`
        f : torch.Tensor
            averaged output function over `n_augs` of shape `(batch, outputs)`
        """
        Js = list()
        for i in range(self.model.output_size):
            def loss_fn(outputs, _):
                return outputs.mean(dim=1)[:, i].sum()

            f = batch_aug_gradient(self.model, loss_fn, x, None, self.kron_jac, 
                                   **self.backward_kwargs).mean(dim=1)
            Js.append(_get_batch_grad(self.model, self.kron_jac))
        Js = torch.stack(Js, dim=1)

        # set gradients to zero, differentiation here only serves Jacobian computation
        self.model.zero_grad()
        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def single_jacobians(self, x, output_ix):
        def loss_fn(outputs, targets):
            if output_ix.ndim == 0:  # scalar case
                return outputs.mean(dim=1)[:, output_ix].sum()
            elif output_ix.ndim == 1:  # vector iid case
                return outputs.mean(dim=1).gather(1, output_ix.unsqueeze(-1)).sum()
            else:
                raise ValueError('output_ix must be scalar or vector')

        f = batch_aug_gradient(self.model, loss_fn, x, None, self.kron_jac, 
                               **self.backward_kwargs).mean(dim=1)
        Js = _get_batch_grad(self.model, self.kron_jac)
        if self.differentiable:
            return Js, f
        return Js.detach(), f.detach()

    def gradients(self, x, y):
        def loss_fn(outputs, targets):
            return self.lossfunc(outputs.mean(dim=1), targets)
        f = batch_aug_gradient(self._model, loss_fn, x, y, self.kron_jac,
                               **self.backward_kwargs).mean(dim=1)
        Gs = _get_batch_grad(self._model, self.kron_jac)
        loss = self.lossfunc(f, y)
        if self.differentiable:
            return Gs, loss
        return Gs.detach(), loss.detach()

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
                kfacs.append([stats.kron.B * stats.kron.A[-1, -1].clone() / M])
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
            raise ValueError('Not supported')
        f, curv = fisher(self._model, self._ggn_type, SHAPE_DIAG, likelihood=self.likelihood,
                         inputs=X, targets=y, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        diag_ggn = curv.matrices_to_vector(None)

        if self.differentiable:
            return self.factor * loss, self.factor * diag_ggn
        return self.factor * loss.detach(), self.factor * diag_ggn.detach()

    def kron(self, X, y, N, **wkwargs):
        if self.last_layer:
            raise ValueError('Not supported')
            f, X = self.model.forward_with_features(X)
        f, curv = fisher(self._model, self._ggn_type, SHAPE_KRON, likelihood=self.likelihood,
                         inputs=X, targets=y, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron.detach()


class AugAsdlGGN(AugAsdlInterface, GGNInterface):
    """Implementation of the `GGNInterface` with Asdl and augmentation support.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True, 
                 kron_jac=False, stochastic=False):
        super().__init__(model, likelihood, last_layer, differentiable, kron_jac)
        self.stochastic = stochastic

    def full(self, x, y, **kwargs):
        """Compute the full GGN \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ggn}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, n_augs, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ggn : torch.Tensor
            GGN `(parameters, parameters)`
        """
        if self.stochastic:
            raise ValueError('Stochastic approximation not implemented for full GGN.')
        if self.last_layer:
            raise ValueError('Not yet tested/implemented for last layer.')

        Js, f = self.jacobians(x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        return loss, H_ggn

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

    def single_diag(self, X, y, output_ix, **kwargs):
        if self.last_layer:
            raise ValueError('Not supported')
        f, curv = fisher(self._model, self._ggn_type, SHAPE_DIAG, likelihood=self.likelihood,
                         inputs=X, targets=y, single_output=output_ix, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        diag_ggn = curv.matrices_to_vector(None)

        if self.differentiable:
            return self.factor * loss, self.factor * diag_ggn
        return self.factor * loss.detach(), self.factor * diag_ggn.detach()

    def single_kron(self, X, y, N, output_ix, **wkwargs):
        if self.last_layer:
            raise ValueError('Not supported')
        f, curv = fisher(self._model, self._ggn_type, SHAPE_KRON, likelihood=self.likelihood,
                         inputs=X, targets=y, single_output=output_ix, **self.backward_kwargs)
        loss = self.lossfunc(f, y)
        M = len(y)
        kron = self._get_kron_factors(curv, M)
        kron = self._rescale_kron_factors(kron, N)

        if self.differentiable:
            return self.factor * loss, self.factor * kron
        return self.factor * loss.detach(), self.factor * kron.detach()


class AugAsdlEF(AugAsdlInterface, EFInterface):
    """Implementation of the `EFInterface` using Asdl and augmentation support.
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
