from functools import partial
import torch
from torch.nn import MSELoss, CrossEntropyLoss

from asdfghjkl.loss import heteroscedastic_mse_loss
from asdfghjkl.kernel import hessian_single_heteroscedastic_regression


class CurvatureInterface:
    """Interface to access curvature for a model and corresponding likelihood.
    A `CurvatureInterface` must inherit from this baseclass and implement the
    necessary functions `jacobians`, `full`, `kron`, and `diag`.
    The interface might be extended in the future to account for other curvature
    structures, for example, a block-diagonal one.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True):
        self.likelihood = likelihood
        self.model = model
        self.last_layer = last_layer
        if likelihood == 'regression':
            self.lossfunc = MSELoss(reduction='sum')
            self.factor = 0.5
        elif likelihood == 'classification':
            self.lossfunc = CrossEntropyLoss(reduction='sum')
            self.factor = 1.
        elif likelihood == 'heteroscedastic_regression':
            self.lossfunc = partial(heteroscedastic_mse_loss, reduction='sum')
            self.factor = 1.
        else:
            raise ValueError('Invalid likelihood')
        self.differentiable = differentiable

    @property
    def _model(self):
        return self.model.last_layer if self.last_layer else self.model

    @property
    def differentiable(self):
        return self._differentiable

    @differentiable.setter
    def differentiable(self, value):
        assert type(value) is bool
        self._differentiable = value
        if value:
            self.backward_kwargs = dict(retain_graph=True, create_graph=True)
        else:
            self.backward_kwargs = dict()

    def jacobians(self, x):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\).

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
        pass

    def single_jacobians(self, x, output_ix):
        """Compute Jacobians \\(\\nabla_\\theta f(x;\\theta)\\) at current parameter \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)` on compatible device with model.
        output_ix : int

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        pass

    @staticmethod
    def last_layer_jacobians(model, x):
        """Compute Jacobians \\(\\nabla_{\\theta_\\textrm{last}} f(x;\\theta_\\textrm{last})\\) 
        only at current last-layer parameter \\(\\theta_{\\textrm{last}}\\).

        Parameters
        ----------
        model : laplace.feature_extractor.FeatureExtractor
        x : torch.Tensor

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, last-layer-parameters, outputs)`
        f : torch.Tensor
            output function `(batch, outputs)`
        """
        f, phi = model.forward_with_features(x)
        bsize = len(x)
        output_size = f.shape[-1]

        # calculate Jacobians using the feature vector 'phi'
        identity = torch.eye(output_size, device=x.device).unsqueeze(0).tile(bsize, 1, 1)
        # Jacobians are batch x output x params
        Js = torch.einsum('kp,kij->kijp', phi, identity).reshape(bsize, output_size, -1)
        if model.last_layer.bias is not None:
            Js = torch.cat([Js, identity], dim=2)

        return Js, f.detach()

    def gradients(self, x, y):
        """Compute gradients \\(\\nabla_\\theta \\ell(f(x;\\theta, y)\\) at current parameter \\(\\theta\\).

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
        pass

    def full(self, x, y, **kwargs):
        """Compute a dense curvature (approximation) in the form of a \\(P \\times P\\) matrix
        \\(H\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            Hessian approximation `(parameters, parameters)`
        """
        pass

    def kron(self, x, y, **kwargs):
        """Compute a Kronecker factored curvature approximation (such as KFAC).
        The approximation to \\(H\\) takes the form of two Kronecker factors \\(Q, H\\),
        i.e., \\(H \\approx Q \\otimes H\\) for each Module in the neural network permitting 
        such curvature.
        \\(Q\\) is quadratic in the input-dimension of a module \\(p_{in} \\times p_{in}\\)
        and \\(H\\) in the output-dimension \\(p_{out} \\times p_{out}\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : `laplace.matrix.Kron`
            Kronecker factored Hessian approximation.
        """

    def diag(self, x, y, **kwargs):
        """Compute a diagonal Hessian approximation to \\(H\\) and is represented as a 
        vector of the dimensionality of parameters \\(\\theta\\).

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H : torch.Tensor
            vector representing the diagonal of H
        """
        pass


class GGNInterface(CurvatureInterface):
    """Generalized Gauss-Newton or Fisher Curvature Interface.
    The GGN is equal to the Fisher information for the available likelihoods.
    In addition to `CurvatureInterface`, methods for Jacobians are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer
    stochastic : bool, default=False
        Fisher if stochastic else GGN
    """
    def __init__(self, model, likelihood, last_layer=False, differentiable=True, stochastic=False):
        self.stochastic = stochastic
        super().__init__(model, likelihood, last_layer, differentiable)

    def _get_full_ggn(self, Js, f, y):
        """Compute full GGN from Jacobians.

        Parameters
        ----------
        Js : torch.Tensor
            Jacobians `(batch, parameters, outputs)`
        f : torch.Tensor
            functions `(batch, outputs)`
        y : torch.Tensor
            labels compatible with loss

        Returns
        -------
        loss : torch.Tensor
        H_ggn : torch.Tensor
            full GGN approximation `(parameters, parameters)`
        """
        loss = self.factor * self.lossfunc(f, y)
        if self.likelihood == 'regression':
            H_ggn = torch.einsum('mkp,mkq->pq', Js, Js)
        elif self.likelihood == 'classification':
            # second derivative of log lik is diag(p) - pp^T
            ps = torch.softmax(f, dim=-1)
            H_lik = torch.diag_embed(ps) - torch.einsum('mk,mc->mck', ps, ps)
            H_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H_lik, Js)
        elif self.likelihood == 'heteroscedastic_regression':
            H_lik = f.new_zeros((f.shape[0], 2, 2))
            eta_1, eta_2 = f[:, 0], f[:, 1]
            H_lik[:, 0, 0] = - 0.5 / eta_2
            H_lik[:, 0, 1] = H_lik[:, 1, 0] = 0.5 * eta_1 / eta_2.square()
            H_lik[:, 1, 1] = 0.5 / eta_2.square() - 0.5 * eta_1.square() / torch.pow(eta_2, 3)
            H_ggn = torch.einsum('mcp,mck,mkq->pq', Js, H_lik, Js)

        if self.differentiable:
            return loss, H_ggn
        return loss.detach(), H_ggn.detach()

    def full(self, x, y, **kwargs):
        """Compute the full GGN \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ggn}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
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
            Js, f = self.last_layer_jacobians(self.model, x)
        else:
            Js, f = self.jacobians(x)
        loss, H_ggn = self._get_full_ggn(Js, f, y)

        if self.differentiable:
            return loss, H_ggn
        return loss.detach(), H_ggn.detach()

    def single_full(self, x, y, output_ix, **kwargs):
        if self.last_layer:
            raise ValueError('Not supported')
        Js, f = self.single_jacobians(x, output_ix)
        loss = self.factor * self.lossfunc(f, y)

        if self.likelihood == 'regression':
            H_ggn = Js.T @ Js
        elif self.likelihood == 'classification':
            p = torch.softmax(f, dim=-1)[:, output_ix]
            hi = p - p.square()
            H_ggn = (Js * hi.unsqueeze(-1)).T @ Js
        elif self.likelihood == 'heteroscedastic_regression':
            l = hessian_single_heteroscedastic_regression(f, output_ix) 
            H_ggn = (Js * l.unsqueeze(-1)).T @ Js
        else:
            raise ValueError('Only supported for regression and classification.')

        if self.differentiable:
            return loss, H_ggn
        return loss.detach(), H_ggn.detach()
        
    def kernel(self, x, y, prec_diag, **kwargs):
        if self.last_layer:
            Js, f = self.last_layer_jacobians(self.model, x)
        else:
            Js, f = self.jacobians(x)
        M, K, P = Js.shape

        if self.likelihood == 'classification':
            p = torch.softmax(f, dim=-1)
            L = torch.diag_embed(p) - torch.einsum('mk,mc->mck', p, p)
            Js_right = (Js.transpose(1, 2) @ L).transpose(1, 2)
        elif self.likelihood == 'regression':
            Js_right = Js
        elif self.likelihood == 'heteroscedastic_regression':
            L = f.new_zeros((f.shape[0], 2, 2))
            eta_1, eta_2 = f[:, 0], f[:, 1]
            L[:, 0, 0] = - 0.5 / eta_2
            L[:, 0, 1] = L[:, 1, 0] = 0.5 * eta_1 / eta_2.square()
            L[:, 1, 1] = 0.5 / eta_2.square() - 0.5 * eta_1.square() / torch.pow(eta_2, 3)
            Js_right = (Js.transpose(1, 2) @ L).transpose(1, 2)

        Js_left = Js.reshape(-1, P) / prec_diag.reshape(1, P)
        K = Js_left @ Js_right.reshape(-1, P).T

        loss = self.factor * self.lossfunc(f, y)
        if self.differentiable:
            return loss, K
        return loss.detach(), K.detach()


class EFInterface(CurvatureInterface):
    """Interface for Empirical Fisher as Hessian approximation.
    In addition to `CurvatureInterface`, methods for gradients are required by subclasses.

    Parameters
    ----------
    model : torch.nn.Module or `laplace.feature_extractor.FeatureExtractor`
        torch model (neural network)
    likelihood : {'classification', 'regression'}
    last_layer : bool, default=False
        only consider curvature of last layer

    Attributes
    ----------
    lossfunc : torch.nn.MSELoss or torch.nn.CrossEntropyLoss
    factor : float
        conversion factor between torch losses and base likelihoods
        For example, \\(\\frac{1}{2}\\) to get to \\(\\mathcal{N}(f, 1)\\) from MSELoss.
    """

    def full(self, x, y, **kwargs):
        """Compute the full EF \\(P \\times P\\) matrix as Hessian approximation
        \\(H_{ef}\\) with respect to parameters \\(\\theta \\in \\mathbb{R}^P\\).
        For last-layer, reduced to \\(\\theta_{last}\\)

        Parameters
        ----------
        x : torch.Tensor
            input data `(batch, input_shape)`
        y : torch.Tensor
            labels `(batch, label_shape)`

        Returns
        -------
        loss : torch.Tensor
        H_ef : torch.Tensor
            EF `(parameters, parameters)`
        """
        Gs, loss = self.gradients(x, y)
        H_ef = Gs.T @ Gs
        if self.differentiable:
            return self.factor * loss, self.factor * H_ef
        return self.factor * loss.detach(), self.factor * H_ef.detach()

    def kernel(self, x, y, prec_diag, **kwargs):
        Gs, loss = self.gradients(x, y)
        M, P = Gs.shape
        K = (Gs / prec_diag.reshape(1, P)) @ Gs.T
        if self.differentiable:
            return self.factor * loss, self.factor * K
        return self.factor * loss.detach(), self.factor * K.detach()
