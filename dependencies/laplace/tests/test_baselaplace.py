import pytest
from itertools import product
import numpy as np
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.distributions import Normal, Categorical
from asdfghjkl.operations import Bias, Scale
from laplace.curvature.asdl import AsdlEF, AsdlGGN
from laplace.curvature.augmented_backpack import AugBackPackGGN
from laplace.curvature.backpack import BackPackEF, BackPackGGN
from laplace.curvature.augmented_asdl import AugAsdlEF, AugAsdlGGN

from laplace.laplace import FullLaplace, KronLaplace, DiagLaplace, BlockDiagLaplace
from laplace.baselaplace import FunctionalLaplace
from tests.utils import jacobians_naive, HetHead


torch.manual_seed(240)
torch.set_default_tensor_type(torch.DoubleTensor)
flavors = [FullLaplace, KronLaplace, DiagLaplace]
backends = [BackPackGGN, BackPackEF]


def get_grad(model):
    return torch.cat([e.grad.flatten() for e in model.parameters()])


@pytest.fixture
def model():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model_single():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.ReLU(), nn.Linear(20, 1))
    setattr(model, 'output_size', 1)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def model_single_fixup():
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.ReLU(), Scale(), Bias(), nn.Linear(20, 1))
    setattr(model, 'output_size', 1)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def class_loader():
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader():
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def reg_loader_single():
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    return DataLoader(TensorDataset(X, y), batch_size=3)


@pytest.fixture
def aug_class_loader():
    X = torch.randn(12, 7, 3)
    y = torch.randint(2, (12,))
    return DataLoader(TensorDataset(X, y), batch_size=3, shuffle=True)


@pytest.fixture
def aug_reg_loader():
    X = torch.randn(12, 7, 3)
    y = torch.randn(12, 2)
    return DataLoader(TensorDataset(X, y), batch_size=3, shuffle=True)


@pytest.fixture
def aug_reg_loader_single():
    X = torch.randn(12, 7, 3)
    y = torch.randn(12, 1)
    return DataLoader(TensorDataset(X, y), batch_size=3, shuffle=True)


@pytest.fixture
def het_model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 2), HetHead())
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def het_reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    return X, y


@pytest.mark.parametrize('backend', backends)
def test_functional_laplace_regression(backend, model):
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    full_lap = FullLaplace(model, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=backend)
    fun_lap = FunctionalLaplace(model, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=backend)
    full_lap.fit(DataLoader(TensorDataset(X, y)))
    fun_lap.fit_batch(X, y, len(y))
    assert torch.allclose(fun_lap.log_marginal_likelihood(), full_lap.log_marginal_likelihood())


def test_functional_indep_laplace_regression(model_single):
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    full_lap = FullLaplace(model_single, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=AsdlGGN)
    fun_lap = FunctionalLaplace(model_single, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=AsdlGGN, 
                                independent=True, backend_kwargs=dict(kron_jac=True))
    full_lap.fit(DataLoader(TensorDataset(X, y)))
    fun_lap.fit_batch(X, y, len(y))
    assert torch.allclose(fun_lap.log_marginal_likelihood(), full_lap.log_marginal_likelihood())


def test_functional_indep_laplace_regression_fixup(model_single_fixup):
    model_single = model_single_fixup
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    full_lap = FullLaplace(model_single, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=AsdlGGN)
    fun_lap = FunctionalLaplace(model_single, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=AsdlGGN, 
                                independent=True, backend_kwargs=dict(kron_jac=True))
    full_lap.fit(DataLoader(TensorDataset(X, y)))
    fun_lap.fit_batch(X, y, len(y))
    assert torch.allclose(fun_lap.log_marginal_likelihood(), full_lap.log_marginal_likelihood())


def test_functional_single_laplace_regression(model_single):
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    fun_lap = FunctionalLaplace(model_single, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=AsdlGGN, independent=True)
    fun_lap_s = FunctionalLaplace(model_single, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=AsdlGGN, 
                                independent=True, single_output=True)
    fun_lap.fit_batch(X, y, len(y))
    fun_lap_s.fit_batch(X, y, len(y))
    assert torch.allclose(fun_lap.log_marginal_likelihood(), fun_lap_s.log_marginal_likelihood())


def test_functional_laplace_hetregression(het_model):
    X = torch.randn(10, 3)
    y = torch.randn(10, 1)
    full_lap = FullLaplace(het_model, 'heteroscedastic_regression', prior_precision=0.3, backend=AsdlGGN)
    fun_lap = FunctionalLaplace(het_model, 'heteroscedastic_regression', prior_precision=0.3, backend=AsdlGGN)
    full_lap.fit(DataLoader(TensorDataset(X, y)))
    fun_lap.fit_batch(X, y, len(y))
    assert torch.allclose(fun_lap.log_marginal_likelihood(), full_lap.log_marginal_likelihood())
    fun_lap = FunctionalLaplace(het_model, 'heteroscedastic_regression', prior_precision=0.3, backend=AsdlGGN, independent=True)
    fun_lap.fit_batch(X, y, len(y))
    assert fun_lap.log_marginal_likelihood() < full_lap.log_marginal_likelihood()
    fun_lap = FunctionalLaplace(het_model, 'heteroscedastic_regression', prior_precision=0.3, backend=AsdlGGN, independent=True, single_output=True)
    fun_lap.fit_batch(X, y, len(y))
    assert fun_lap.log_marginal_likelihood() < full_lap.log_marginal_likelihood()


def test_functional_laplace_restriction(model):
    with pytest.raises(ValueError):
        FunctionalLaplace(model, 'classification', independent=False, single_output=True)


@pytest.mark.parametrize('backend', backends)
def test_functional_laplace_classification(backend, model):
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    full_lap = FullLaplace(model, 'classification', prior_precision=0.3, backend=backend)
    fun_lap = FunctionalLaplace(model, 'classification', prior_precision=0.3, backend=backend)
    full_lap.fit(DataLoader(TensorDataset(X, y)))
    fun_lap.fit_batch(X, y, len(y))
    assert torch.allclose(fun_lap.log_marginal_likelihood(), full_lap.log_marginal_likelihood())

    
@pytest.mark.parametrize('backend', backends)
def test_functional_laplace_additivity_regression(backend, model):
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    fun_lap = FunctionalLaplace(model, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=backend, sod=True)
    fun_lap.fit_batch(X, y, 20)
    fun_lap_add = FunctionalLaplace(model, 'regression', sigma_noise=1.8, prior_precision=0.3, backend=backend)
    fun_lap_add.fit_batch(X, y, 20)
    fun_lap_add.fit_batch(X, y, 20)
    assert torch.allclose(fun_lap.log_likelihood, fun_lap_add.log_likelihood)
    assert torch.allclose(fun_lap.log_marginal_likelihood(), fun_lap_add.log_marginal_likelihood())


@pytest.mark.parametrize('backend', backends)
def test_functional_laplace_additivity_classification(backend, model):
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    fun_lap = FunctionalLaplace(model, 'classification', prior_precision=0.3, backend=backend, sod=True)
    fun_lap.fit_batch(X, y, 20)
    fun_lap_add = FunctionalLaplace(model, 'classification', prior_precision=0.3, backend=backend)
    fun_lap_add.fit_batch(X, y, 20)
    fun_lap_add.fit_batch(X, y, 20)
    assert torch.allclose(fun_lap.log_marginal_likelihood(), fun_lap_add.log_marginal_likelihood())


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init(laplace, model):
    lap = laplace(model, 'classification')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_invalid_likelihood(laplace, model):
    with pytest.raises(ValueError):
        lap = laplace(model, 'otherlh')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_noise(laplace, model):
    # float
    sigma_noise = 1.2
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)
    # torch.tensor 0-dim
    sigma_noise = torch.tensor(1.2)
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)
    # torch.tensor 1-dim
    sigma_noise = torch.tensor(1.2).reshape(-1)
    lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)

    # for classification should fail
    sigma_noise = 1.2
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='classification', sigma_noise=sigma_noise)

    # other than that should fail
    # higher dim
    sigma_noise = torch.tensor(1.2).reshape(1, 1)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)
    # other datatype, only reals supported
    sigma_noise = '1.2'
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', sigma_noise=sigma_noise)


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_precision(laplace, model):
    # float
    precision = 10.6
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 0-dim
    precision = torch.tensor(10.6)
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 1-dim
    precision = torch.tensor(10.7).reshape(-1)
    lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 1-dim param-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_params)
    if laplace == KronLaplace:
        # Kron should not accept per parameter prior precision
        with pytest.raises(ValueError):
            lap = laplace(model, likelihood='regression', prior_precision=precision)
    else:
        lap = laplace(model, likelihood='regression', prior_precision=precision)
    # torch.tensor 1-dim layer-shape
    precision = torch.tensor(10.7).reshape(-1).repeat(model.n_layers)
    lap = laplace(model, likelihood='regression', prior_precision=precision)

    # other than that should fail
    # higher dim
    precision = torch.tensor(10.6).reshape(1, 1)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision)
    # unmatched dim
    precision = torch.tensor(10.6).reshape(-1).repeat(17)
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision)
    # other datatype, only reals supported
    precision = '1.5'
    with pytest.raises(ValueError):
        lap = laplace(model, likelihood='regression', prior_precision=precision)


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_prior_mean_and_scatter(laplace, model):
    mean = parameters_to_vector(model.parameters())
    P = len(mean)
    lap_scalar_mean = laplace(model, 'classification',
                              prior_precision=1e-2, prior_mean=1.)
    assert torch.allclose(lap_scalar_mean.prior_mean, torch.tensor([1.]))
    lap_tensor_mean = laplace(model, 'classification',
                              prior_precision=1e-2, prior_mean=torch.ones(1))
    assert torch.allclose(lap_tensor_mean.prior_mean, torch.tensor([1.]))
    lap_tensor_scalar_mean = laplace(model, 'classification',
                                     prior_precision=1e-2, prior_mean=torch.ones(1)[0])
    assert torch.allclose(lap_tensor_scalar_mean.prior_mean, torch.tensor(1.))
    lap_tensor_full_mean = laplace(model, 'classification',
                                   prior_precision=1e-2, prior_mean=torch.ones(P))
    assert torch.allclose(lap_tensor_full_mean.prior_mean, torch.ones(P))
    expected = ((mean - 1) * 1e-2) @ (mean - 1)
    assert expected.ndim == 0
    assert torch.allclose(lap_scalar_mean.scatter, expected)
    assert lap_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_mean.scatter, expected)
    assert lap_tensor_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_scalar_mean.scatter, expected)
    assert lap_tensor_scalar_mean.scatter.shape == expected.shape
    assert torch.allclose(lap_tensor_full_mean.scatter, expected)
    assert lap_tensor_full_mean.scatter.shape == expected.shape

    # too many dims
    with pytest.raises(ValueError):
        prior_mean = torch.ones(P).unsqueeze(-1)
        laplace(model, 'classification', prior_precision=1e-2, prior_mean=prior_mean)

    # unmatched dim
    with pytest.raises(ValueError):
        prior_mean = torch.ones(P-3)
        laplace(model, 'classification', prior_precision=1e-2, prior_mean=prior_mean)

    # invalid argument type
    with pytest.raises(ValueError):
        laplace(model, 'classification', prior_precision=1e-2, prior_mean='72')


@pytest.mark.parametrize('laplace', flavors)
def test_laplace_init_temperature(laplace, model):
    # valid float
    T = 1.1
    lap = laplace(model, likelihood='classification', temperature=T)
    assert lap.temperature == T


@pytest.mark.parametrize('laplace,lh', product(flavors, ['classification', 'regression']))
def test_laplace_functionality(laplace, lh, model, reg_loader, class_loader):
    if lh == 'classification':
        loader = class_loader
        sigma_noise = 1.
    else:
        loader = reg_loader
        sigma_noise = 0.3
    lap = laplace(model, lh, sigma_noise=sigma_noise, prior_precision=0.7)
    lap.fit(loader)
    assert lap.n_data == len(loader.dataset)
    assert lap.n_outputs == model.output_size
    f = model(loader.dataset.tensors[0])
    y = loader.dataset.tensors[1]
    assert f.shape == torch.Size([10, 2])

    # Test log likelihood (Train)
    log_lik = lap.log_likelihood
    # compute true log lik
    if lh == 'classification':
        log_lik_true = Categorical(logits=f).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)
    else:
        assert y.size() == f.size()
        log_lik_true = Normal(loc=f, scale=sigma_noise).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)
        # change likelihood and test again
        lap.sigma_noise = 0.72
        log_lik = lap.log_likelihood
        log_lik_true = Normal(loc=f, scale=0.72).log_prob(y).sum()
        assert torch.allclose(log_lik, log_lik_true)

    # Test marginal likelihood
    # lml = log p(y|f) - 1/2 theta @ prior_prec @ theta
    #       + 1/2 logdet prior_prec - 1/2 log det post_prec
    lml = log_lik_true
    theta = parameters_to_vector(model.parameters()).detach()
    assert torch.allclose(theta, lap.mean)
    prior_prec = torch.diag(lap.prior_precision_diag)
    assert prior_prec.shape == torch.Size([len(theta), len(theta)])
    lml = lml - 1/2 * theta @ prior_prec @ theta
    Sigma_0 = torch.inverse(prior_prec)
    if laplace == DiagLaplace:
        log_det_post_prec = lap.posterior_precision.log().sum()
    else:
        log_det_post_prec = lap.posterior_precision.logdet()
    lml = lml + 1/2 * (prior_prec.logdet() - log_det_post_prec)
    assert torch.allclose(lml, lap.log_marginal_likelihood())

    # test sampling
    torch.manual_seed(61)
    samples = lap.sample(n_samples=1)
    assert samples.shape == torch.Size([1, len(theta)])
    samples = lap.sample(n_samples=1000000)
    assert samples.shape == torch.Size([1000000, len(theta)])
    mu_comp = samples.mean(dim=0)
    mu_true = lap.mean
    assert torch.allclose(mu_comp, mu_true, rtol=1, atol=1e-3)

    # test functional variance
    if laplace == FullLaplace:
        Sigma = lap.posterior_covariance
    elif laplace == KronLaplace:
        Sigma = lap.posterior_precision.to_matrix(exponent=-1)
    elif laplace == DiagLaplace:
        Sigma = torch.diag(lap.posterior_variance)
    Js, f = jacobians_naive(model, loader.dataset.tensors[0])
    true_f_var = torch.einsum('mkp,pq,mcq->mkc', Js, Sigma, Js)
    comp_f_var = lap.functional_variance(Js)
    assert torch.allclose(true_f_var, comp_f_var, rtol=1e-4)


@pytest.mark.parametrize('laplace', flavors)
def test_regression_predictive(laplace, model, reg_loader):
    lap = laplace(model, 'regression', sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive, functional variance tested already above.
    f_mu, f_var = lap(X, pred_type='glm')
    assert torch.allclose(f_mu, f)
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1], f_mu.shape[1]])
    assert len(f_mu) == len(X)

    # NN predictive (only diagonal variance estimation)
    f_mu, f_var = lap(X, pred_type='nn')
    assert f_mu.shape == f_var.shape
    assert f_var.shape == torch.Size([f_mu.shape[0], f_mu.shape[1]])
    assert len(f_mu) == len(X)


@pytest.mark.parametrize('laplace', flavors)
def test_classification_predictive(laplace, model, class_loader):
    lap = laplace(model, 'classification', prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive
    f_pred = lap(X, pred_type='glm', link_approx='mc', n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
    f_pred = lap(X, pred_type='glm', link_approx='probit')
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1
    f_pred = lap(X, pred_type='glm', link_approx='bridge')
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1


    # NN predictive
    f_pred = lap(X, pred_type='nn', n_samples=100)
    assert f_pred.shape == f.shape
    assert torch.allclose(f_pred.sum(), torch.tensor(len(f_pred), dtype=torch.double))  # sum up to 1


@pytest.mark.parametrize('laplace', flavors)
def test_regression_predictive_samples(laplace, model, reg_loader):
    lap = laplace(model, 'regression', sigma_noise=0.3, prior_precision=0.7)
    lap.fit(reg_loader)
    X, y = reg_loader.dataset.tensors
    f = model(X)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive, functional variance tested already above.
    fsamples = lap.predictive_samples(X, pred_type='glm', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])

    # NN predictive (only diagonal variance estimation)
    fsamples = lap.predictive_samples(X, pred_type='nn', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])


@pytest.mark.parametrize('laplace', flavors)
def test_classification_predictive_samples(laplace, model, class_loader):
    lap = laplace(model, 'classification', prior_precision=0.7)
    lap.fit(class_loader)
    X, y = class_loader.dataset.tensors
    f = torch.softmax(model(X), dim=-1)

    # error
    with pytest.raises(ValueError):
        lap(X, pred_type='linear')

    # GLM predictive
    fsamples = lap.predictive_samples(X, pred_type='glm', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1

    # NN predictive
    f_pred = lap.predictive_samples(X, pred_type='nn', n_samples=100)
    assert fsamples.shape == torch.Size([100, f.shape[0], f.shape[1]])
    assert np.allclose(fsamples.sum().item(), len(f) * 100)  # sum up to 1

    
@pytest.mark.parametrize('kron_jac', [True, False])
def test_marglik_indep_kernel(model_single, reg_loader_single, kron_jac):
    prior_prec = 0.089
    sigma_noise = 0.27
    backend_kwargs = dict(differentiable=True, kron_jac=kron_jac)
    # control with full laplace
    full_lap = FullLaplace(model_single, 'regression', backend=AsdlGGN, 
                           backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                           sigma_noise=sigma_noise)
    full_lap.fit(reg_loader_single)
    model_single.zero_grad()
    marglik = full_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model_single).clone()
    # compare to kernel laplace
    kernel_lap = FunctionalLaplace(model_single, 'regression', backend=AsdlGGN, 
                                   backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                                   sigma_noise=sigma_noise, independent=True)
    # need  full batch
    kernel_lap.fit(DataLoader(reg_loader_single.dataset, batch_size=len(reg_loader_single.dataset)))
    model_single.zero_grad()
    kernel_marglik = kernel_lap.log_marginal_likelihood()
    kernel_marglik.backward()
    kernel_grad = get_grad(model_single).clone()
    assert torch.allclose(marglik, kernel_marglik)
    assert torch.allclose(grad, kernel_grad)

    
@pytest.mark.parametrize('lh', ['classification', 'regression'])
def test_marglik_indep_kernel_kron_jac(model, reg_loader, class_loader, lh):
    loader = reg_loader if lh == 'regression' else class_loader
    prior_prec = 2/3
    sigma_noise = 0.3 if lh == 'regression' else 1.0
    # for linear model both should be equivalent
    # with kron_jac
    backend_kwargs = dict(differentiable=True, kron_jac=True)
    kernel_lap = FunctionalLaplace(model, lh, backend=AsdlGGN, 
                                   backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                                   sigma_noise=sigma_noise, independent=True)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    model.zero_grad()
    marglik_kron = kernel_lap.log_marginal_likelihood()
    marglik_kron.backward()
    grad_kron = get_grad(model).clone()
    # without kron_jac
    backend_kwargs = dict(differentiable=True, kron_jac=False)
    kernel_lap = FunctionalLaplace(model, lh, backend=AsdlGGN, 
                                   backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                                   sigma_noise=sigma_noise, independent=True)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    model.zero_grad()
    marglik = kernel_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model).clone()
    assert torch.allclose(marglik, marglik_kron)
    assert torch.allclose(grad, grad_kron)
    

@pytest.mark.parametrize('backend,lh', 
                         product([AsdlGGN, AsdlEF, BackPackGGN, BackPackEF], 
                                 ['classification', 'regression']))
def test_marglik_kernel_vs_full(model, backend, lh, class_loader, reg_loader):
    loader = reg_loader if lh == 'regression' else class_loader
    prior_prec = 2/3
    sigma_noise = 0.3 if lh == 'regression' else 1.0
    backend_kwargs = dict(differentiable=True)
    # control with full laplace
    full_lap = FullLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                           prior_precision=prior_prec, sigma_noise=sigma_noise)
    full_lap.fit(loader)
    model.zero_grad()
    marglik = full_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model).clone()
    # compare to kernel laplace
    kernel_lap = FunctionalLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                                   prior_precision=prior_prec, sigma_noise=sigma_noise)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    model.zero_grad()
    kernel_marglik = kernel_lap.log_marginal_likelihood()
    kernel_marglik.backward()
    kernel_grad = get_grad(model).clone()
    assert torch.allclose(marglik, kernel_marglik)
    assert torch.allclose(grad, kernel_grad)


@pytest.mark.parametrize('backend,lh', product([AsdlGGN, AsdlEF], ['classification', 'regression']))
def test_marglik_bound_ordering(model, backend, lh, class_loader, reg_loader):
    loader = reg_loader if lh == 'regression' else class_loader
    prior_prec = 2/3
    sigma_noise = 0.3 if lh == 'regression' else 1.0
    backend_kwargs = dict(differentiable=False)
    # control with full laplace
    full_lap = FullLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                           prior_precision=prior_prec, sigma_noise=sigma_noise)
    full_lap.fit(loader)
    model.zero_grad()
    marglik = full_lap.log_marginal_likelihood()

    # control functional
    kernel_lap = FunctionalLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                                   prior_precision=prior_prec, sigma_noise=sigma_noise)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    kernel_marglik = kernel_lap.log_marginal_likelihood()
    assert torch.allclose(marglik, kernel_marglik)

    ## parametric bounds
    # blkdiag should be lower
    blk_lap = BlockDiagLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                               prior_precision=prior_prec, sigma_noise=sigma_noise)
    blk_lap.fit(loader)
    model.zero_grad()
    marglik_blk = blk_lap.log_marginal_likelihood()
    assert marglik_blk < marglik
    # kron should be lower
    kron_lap = KronLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                           prior_precision=prior_prec, sigma_noise=sigma_noise)
    kron_lap.fit(loader)
    model.zero_grad()
    marglik_kron = kron_lap.log_marginal_likelihood()
    assert marglik_kron <= marglik
    # diag should be even lower
    diag_lap = DiagLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                           prior_precision=prior_prec, sigma_noise=sigma_noise)
    diag_lap.fit(loader)
    model.zero_grad()
    marglik_diag = diag_lap.log_marginal_likelihood()
    assert marglik_diag <= marglik_kron


@pytest.mark.parametrize('backend,lap', 
                         product([AsdlGGN, AsdlEF], 
                                 [FullLaplace, FunctionalLaplace, KronLaplace, DiagLaplace]))
def test_marglik_sod_bounds(model, backend, lap, class_loader):
    lh = 'classification'
    loader = class_loader
    prior_prec = 2/3
    backend_kwargs = dict(differentiable=False)
    # full data
    full_loader = DataLoader(loader.dataset, batch_size=len(loader.dataset))
    la = lap(model, lh, backend=backend, backend_kwargs=backend_kwargs, 
             prior_precision=prior_prec)
    la.fit(full_loader)
    marglik = la.log_marginal_likelihood().item()

    # sod
    N = len(loader.dataset)
    sodloader = DataLoader(Subset(loader.dataset, range(int(N/2))), batch_size=int(N/2))
    sodloader2 = DataLoader(Subset(loader.dataset, range(int(N/2), N)), batch_size=int(N/2))
    la = lap(model, lh, backend=backend, backend_kwargs=backend_kwargs, 
             prior_precision=prior_prec, sod=True)
    la.fit(sodloader)
    la.n_data = N
    marglik_sod_a = la.log_marginal_likelihood().item()
    la.fit(sodloader2)
    la.n_data = N
    marglik_sod_b = la.log_marginal_likelihood().item()
    marglik_sod = (marglik_sod_a + marglik_sod_b) / 2
    assert marglik_sod <= marglik

    if backend == AsdlEF:
        return
    # sod single output for ggn
    if lap == FunctionalLaplace:
        la = lap(model, lh, backend=backend, backend_kwargs=backend_kwargs, 
                prior_precision=prior_prec, sod=True, single_output=True, independent=True)
    else:
        la = lap(model, lh, backend=backend, backend_kwargs=backend_kwargs, 
                prior_precision=prior_prec, sod=True, single_output=True)
    marglik_sod_single = 0
    torch.manual_seed(1)
    for loader in [sodloader, sodloader2]:
        la.fit(loader)
        la.n_data = N
        marglik_sod_single += la.log_marginal_likelihood().item() / 2
    assert marglik_sod_single < marglik_sod

    
@pytest.mark.parametrize('backend,lh', 
                         product([AsdlGGN, AsdlEF, BackPackGGN, BackPackEF], 
                                 ['classification', 'regression']))
def test_marglik_kernel_vs_full_sod(model, backend, lh, class_loader, reg_loader):
    loader = reg_loader if lh == 'regression' else class_loader
    prior_prec = 2/3
    sigma_noise = 0.3 if lh == 'regression' else 1.0
    backend_kwargs = dict(differentiable=True)
    # control with full laplace
    full_lap = FullLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                           prior_precision=prior_prec, sigma_noise=sigma_noise, sod=True)
    full_lap.fit(loader)
    # as if the dataset has 2x more data but loader only holds half
    full_lap.n_data *= 2
    model.zero_grad()
    marglik = full_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model).clone()
    # compare to kernel laplace
    kernel_lap = FunctionalLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                                   prior_precision=prior_prec, sigma_noise=sigma_noise, sod=True)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    kernel_lap.n_data *= 2  
    model.zero_grad()
    kernel_marglik = kernel_lap.log_marginal_likelihood()
    kernel_marglik.backward()
    kernel_grad = get_grad(model).clone()
    assert torch.allclose(marglik, kernel_marglik)
    assert torch.allclose(grad, kernel_grad)



@pytest.mark.parametrize('backend,lh', 
                         product([AugAsdlGGN, AugAsdlEF, AugBackPackGGN],
                                 ['classification', 'regression']))
def test_marglik_kernel_vs_full_aug(model, backend, lh, aug_class_loader, aug_reg_loader):
    loader = aug_reg_loader if lh == 'regression' else aug_class_loader
    prior_prec = 0.7
    sigma_noise = 0.3 if lh == 'regression' else 1.0
    backend_kwargs = dict(differentiable=True)
    # control with full laplace
    full_lap = FullLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                           prior_precision=prior_prec, sigma_noise=sigma_noise)
    full_lap.fit(loader)
    model.zero_grad()
    marglik = full_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model).clone()
    # compare to kernel laplace
    kernel_lap = FunctionalLaplace(model, lh, backend=backend, backend_kwargs=backend_kwargs,
                                   prior_precision=prior_prec, sigma_noise=sigma_noise)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    model.zero_grad()
    kernel_marglik = kernel_lap.log_marginal_likelihood()
    kernel_marglik.backward()
    kernel_grad = get_grad(model).clone()
    assert torch.allclose(marglik, kernel_marglik)
    assert torch.allclose(grad, kernel_grad)


@pytest.mark.parametrize('kron_jac', [True, False])
def test_marglik_indep_kernel_aug(model_single, aug_reg_loader_single, kron_jac):
    prior_prec = 0.089
    sigma_noise = 0.27
    backend_kwargs = dict(differentiable=True, kron_jac=kron_jac)
    # control with full laplace
    full_lap = FullLaplace(model_single, 'regression', backend=AugAsdlGGN, 
                           backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                           sigma_noise=sigma_noise)
    full_lap.fit(aug_reg_loader_single)
    model_single.zero_grad()
    marglik = full_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model_single).clone()
    # compare to kernel laplace
    kernel_lap = FunctionalLaplace(model_single, 'regression', backend=AugAsdlGGN, 
                                   backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                                   sigma_noise=sigma_noise, independent=False)
    # need  full batch
    kernel_lap.fit(DataLoader(aug_reg_loader_single.dataset, batch_size=len(aug_reg_loader_single.dataset)))
    model_single.zero_grad()
    kernel_marglik = kernel_lap.log_marginal_likelihood()
    kernel_marglik.backward()
    kernel_grad = get_grad(model_single).clone()
    assert torch.allclose(marglik, kernel_marglik)
    assert torch.allclose(grad, kernel_grad, rtol=1e-2)


@pytest.mark.parametrize('lh', ['classification', 'regression'])
def test_marglik_indep_kernel_kron_jac_aug(model, aug_reg_loader, aug_class_loader, lh):
    loader = aug_reg_loader if lh == 'regression' else aug_class_loader
    prior_prec = torch.exp(torch.linspace(-1, 1, steps=len(list(model.parameters()))))
    sigma_noise = 0.27 if lh == 'regression' else 1.0
    # for linear model both should be equivalent
    # with kron_jac
    backend_kwargs = dict(differentiable=True, kron_jac=True)
    kernel_lap = FunctionalLaplace(model, lh, backend=AugAsdlGGN, 
                                   backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                                   sigma_noise=sigma_noise, independent=True)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    model.zero_grad()
    marglik_kron = kernel_lap.log_marginal_likelihood()
    marglik_kron.backward()
    grad_kron = get_grad(model).clone()
    # without kron_jac
    backend_kwargs = dict(differentiable=True, kron_jac=False)
    kernel_lap = FunctionalLaplace(model, lh, backend=AugAsdlGGN, 
                                   backend_kwargs=backend_kwargs, prior_precision=prior_prec, 
                                   sigma_noise=sigma_noise, independent=True)
    # need  full batch
    kernel_lap.fit(DataLoader(loader.dataset, batch_size=len(loader.dataset)))
    model.zero_grad()
    marglik = kernel_lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model).clone()
    assert torch.allclose(marglik, marglik_kron)
    assert torch.allclose(grad, grad_kron)


@pytest.mark.parametrize('curv_type,laplace', 
                         product(['ggn', 'ef'], [FullLaplace, DiagLaplace, KronLaplace, FunctionalLaplace]))
def test_differentiable_marglik_backends_class(laplace, model, class_loader, curv_type):
    if curv_type == 'ef' and laplace is KronLaplace:
        # not to be tested since backpack doesn't have Kron-EF
        return
    if curv_type == 'ggn':
        ba, bb = AsdlGGN, BackPackGGN
    else:
        ba, bb = AsdlEF, BackPackEF
    backend_kwargs = dict(differentiable=True)

    lap = laplace(model, 'classification', backend=ba, backend_kwargs=backend_kwargs)
    lap.fit(class_loader)
    model.zero_grad()
    marglik = lap.log_marginal_likelihood()
    marglik.backward()
    grad = get_grad(model).clone()

    lap = laplace(model, 'classification', backend=bb, backend_kwargs=backend_kwargs)
    lap.fit(class_loader)
    model.zero_grad()
    marglikb = lap.log_marginal_likelihood()
    marglikb.backward()
    gradb = get_grad(model).clone()

    assert torch.allclose(marglik, marglikb)
    # if not (curv_type == 'ggn' and laplace in [DiagLaplace, KronLaplace]):
    assert torch.allclose(grad, gradb)


@pytest.mark.parametrize('backend', [AsdlGGN, BackPackGGN])
def test_differentiable_marglik_diag(model, class_loader, backend):
    backend_kwargs = dict(differentiable=True)
    lap = FullLaplace(model, 'classification', backend=backend, backend_kwargs=backend_kwargs)
    lap.fit(class_loader)
    model.zero_grad()
    diag_posterior_prec = lap.posterior_precision.diagonal()
    pps = diag_posterior_prec.sum()
    pps.backward()
    grad = get_grad(model).clone()

    lap = DiagLaplace(model, 'classification', backend=backend, backend_kwargs=backend_kwargs)
    lap.fit(class_loader)
    model.zero_grad()
    ppsb = lap.posterior_precision.sum()
    ppsb.backward()
    gradb = get_grad(model).clone()

    assert torch.allclose(pps, ppsb)
    assert torch.allclose(grad, gradb)


@pytest.mark.parametrize('single_output', [False, True])
def test_kernel_marglik_vs_full(model, class_loader, single_output):
    loader = class_loader
    backend_kwargs = dict(differentiable=False)
    prior_prec = 2/3
    lap = FullLaplace(model, 'classification', backend=AsdlGGN, prior_precision=prior_prec,
                      backend_kwargs=backend_kwargs, sod=True, single_output=single_output)
    lapk = FunctionalLaplace(model, 'classification', backend=AsdlGGN, prior_precision=prior_prec,
                             backend_kwargs=backend_kwargs, sod=True, single_output=single_output,
                             independent=single_output)

    N = len(loader.dataset)
    sodloader = DataLoader(Subset(loader.dataset, range(int(N/2))), batch_size=int(N/2))
    model.zero_grad()
    torch.manual_seed(711)  # ensures same random_ix in closures
    lap.fit(sodloader)
    lap.n_data = N
    marglik_lap = lap.log_marginal_likelihood()
    torch.manual_seed(711)  # ensures same random_ix in closures
    lapk.fit(sodloader)
    lapk.n_data = N
    marglik_lapk = lapk.log_marginal_likelihood()
    assert torch.allclose(marglik_lapk, marglik_lap)
