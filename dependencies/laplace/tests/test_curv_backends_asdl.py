import pytest
import torch
from torch import nn
from torch.nn.utils import parameters_to_vector

from asdfghjkl.operations import Bias, Scale

from laplace.curvature import AsdlGGN, AsdlEF, BackPackGGN, BackPackEF
from tests.utils import jacobians_naive, HetHead


@pytest.fixture
def model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Linear(3, 20), nn.Tanh(), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def linear_model():
    model = torch.nn.Sequential(nn.Linear(3, 2, bias=False))
    setattr(model, 'output_size', 2)
    return model


@pytest.fixture
def class_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randint(2, (10,))
    return X, y


@pytest.fixture
def reg_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3)
    y = torch.randn(10, 2)
    return X, y


@pytest.fixture
def complex_model():
    torch.manual_seed(711)
    model = torch.nn.Sequential(nn.Conv2d(3, 4, 2, 2), nn.Flatten(), nn.Tanh(),
                                nn.Linear(16, 20), nn.Tanh(), Scale(), Bias(), nn.Linear(20, 2))
    setattr(model, 'output_size', 2)
    model_params = list(model.parameters())
    setattr(model, 'n_layers', len(model_params))  # number of parameter groups
    setattr(model, 'n_params', len(parameters_to_vector(model_params)))
    return model


@pytest.fixture
def complex_class_Xy():
    torch.manual_seed(711)
    X = torch.randn(10, 3, 5, 5)
    y = torch.randint(2, (10,))
    return X, y


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

    
@pytest.mark.parametrize('output_ix', [torch.tensor(0), torch.tensor(1)])
def test_diag_single(class_Xy, model, output_ix):
    X, y = class_Xy
    backend = AsdlGGN(model, 'classification')
    loss, dggn = backend.single_diag(X, y, output_ix)
    Js, f = jacobians_naive(model, X)
    Jsi = Js[:, output_ix, :].squeeze()
    p = torch.softmax(f, dim=1)[:, output_ix].squeeze()
    h = p - p.square()
    dggn_true = torch.diag((Jsi * h.unsqueeze(1)).T @ Jsi)
    assert torch.allclose(dggn_true, dggn)


@pytest.mark.parametrize('output_ix', [torch.tensor(0), torch.tensor(1)])
def test_kron_single(class_Xy, linear_model, output_ix):
    X, y = class_Xy
    X = torch.ones_like(X)
    model = linear_model
    backend = AsdlGGN(model, 'classification')
    loss, kron = backend.single_kron(X, y, len(y), output_ix)
    Js, f = jacobians_naive(model, X)
    Jsi = Js[:, output_ix, :].squeeze()
    p = torch.softmax(f, dim=1)[:, output_ix].squeeze()
    h = p - p.square()
    ggn_true = (Jsi * h.unsqueeze(1)).T @ Jsi
    ggn_approx = kron.to_matrix()
    assert torch.allclose(ggn_true, ggn_approx)
    

def test_diag_ggn_cls_against_backpack_full(class_Xy, model):
    X, y = class_Xy
    backend = AsdlGGN(model, 'classification', stochastic=False)
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackGGN(model, 'classification', stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ef_cls_against_backpack_full(class_Xy, model):
    X, y = class_Xy
    backend = AsdlEF(model, 'classification')
    loss, dggn = backend.diag(X[:5], y[:5])
    loss2, dggn2 = backend.diag(X[5:], y[5:])
    loss += loss2
    dggn += dggn2

    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = BackPackEF(model, 'classification')
    loss_f, H_ggn = backend.full(X, y)
    assert torch.allclose(loss, loss_f)
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_diag_ggn_reg_asdl(reg_Xy, model):
    X, y = reg_Xy
    backend = AsdlGGN(model, 'regression', stochastic=False)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # check against manually computed full GGN:
    backend = AsdlGGN(model, 'regression', stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert loss == loss_f
    print(dggn[:5], H_ggn.diagonal()[:5])
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_full_vs_diag_ef_reg_asdl(reg_Xy, model):
    X, y = reg_Xy
    backend = AsdlEF(model, 'regression')
    loss, diag_ef = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(diag_ef) == model.n_params

    # check against manually computed full GGN:
    backend = AsdlEF(model, 'regression')
    loss_f, H_ef = backend.full(X, y)
    assert loss == loss_f
    assert torch.allclose(diag_ef, H_ef.diagonal())


def test_diag_ggn_stoch_cls(class_Xy, model):
    X, y = class_Xy
    backend = AsdlGGN(model, 'classification', stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params

    # same order of magnitude os non-stochastic.
    backend = AsdlGGN(model, 'classification', stochastic=False)
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1e1)


@pytest.mark.parametrize('Backend', [AsdlEF, AsdlGGN])
def test_kron_vs_diag_class(class_Xy, model, Backend):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = class_Xy
    backend = Backend(model, 'classification')
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


@pytest.mark.parametrize('Backend', [AsdlEF, AsdlGGN])
def test_kron_batching_correction(class_Xy, model, Backend):
    X, y = class_Xy
    backend = Backend(model, 'classification')
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == model.n_params

    N = len(X)
    M = 3
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


def test_kron_batching_correction_single(class_Xy, model):
    X, y = class_Xy
    backend = AsdlGGN(model, 'classification')
    loss, kron = backend.single_kron(X, y, N=len(X), output_ix=torch.tensor(0))
    assert len(kron.diag()) == model.n_params

    N = len(X)
    M = 3
    loss1, kron1 = backend.single_kron(X[:M], y[:M], N=N, output_ix=torch.tensor(0))
    loss2, kron2 = backend.single_kron(X[M:], y[M:], N=N, output_ix=torch.tensor(0))
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)


@pytest.mark.parametrize('Backend', [AsdlGGN, AsdlEF])
def test_kron_summing_up_vs_diag(class_Xy, model, Backend):
    # For a single data point, Kron is exact and should equal diag class_Xy
    X, y = class_Xy
    backend = Backend(model, 'classification')
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


def test_complex_diag_ggn_stoch_cls(complex_class_Xy, complex_model):
    X, y = complex_class_Xy
    backend = AsdlGGN(complex_model, 'classification', stochastic=True)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params

    # same order of magnitude os non-stochastic.
    loss_ns, dggn_ns = backend.diag(X, y)
    assert loss_ns == loss
    assert torch.allclose(dggn, dggn_ns, atol=1e-8, rtol=1)


@pytest.mark.parametrize('Backend', [AsdlEF, AsdlGGN])
def test_complex_kron_vs_diag(complex_class_Xy, complex_model, Backend):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = complex_class_Xy
    backend = Backend(complex_model, 'classification')
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == complex_model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-1)


@pytest.mark.parametrize('Backend', [AsdlEF, AsdlGGN])
def test_complex_kron_batching_correction(complex_class_Xy, complex_model, Backend):
    X, y = complex_class_Xy
    backend = Backend(complex_model, 'classification')
    loss, kron = backend.kron(X, y, N=len(X))
    assert len(kron.diag()) == complex_model.n_params

    N = len(X)
    M = 3
    loss1, kron1 = backend.kron(X[:M], y[:M], N=N)
    loss2, kron2 = backend.kron(X[M:], y[M:], N=N)
    kron_two = kron1 + kron2
    loss_two = loss1 + loss2
    assert torch.allclose(kron.diag(), kron_two.diag())
    assert torch.allclose(loss, loss_two)

@pytest.mark.parametrize('Backend', [AsdlEF, AsdlGGN])
def test_complex_kron_batching_correction(complex_class_Xy, complex_model, Backend):
    X, y =complex_class_Xy
    backend = Backend(complex_model, 'classification')
    loss, diag = backend.diag(X, y)
    assert len(diag) == complex_model.n_params

    N = len(X)
    M = 3
    loss1, diag1 = backend.diag(X[:M], y[:M])
    loss2, diag2 = backend.diag(X[M:], y[M:])
    diag_two = diag1 + diag2
    loss_two = loss1 + loss2
    assert torch.allclose(diag, diag_two)
    assert torch.allclose(loss, loss_two)

@pytest.mark.parametrize('Backend', [AsdlGGN, AsdlEF])
def test_complex_kron_summing_up_vs_diag_class(complex_class_Xy, complex_model, Backend):
    # For a single data point, Kron is exact and should equal diag class_Xy
    X, y = complex_class_Xy
    backend = Backend(complex_model, 'classification')
    loss, dggn = backend.diag(X, y, N=len(X))
    loss, kron = backend.kron(X, y, N=len(X))
    assert torch.allclose(kron.diag().norm(), dggn.norm(), rtol=1e-2)


def test_kron_normalization_ggn_class(class_Xy, model):
    X, y = class_Xy
    xi, yi = X[:1], y[:1]
    backend = AsdlGGN(model, 'classification', stochastic=False)
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test  = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)


def test_kron_normalization_ef_class(class_Xy, model):
    X, y = class_Xy
    xi, yi = X[:1], y[:1]
    backend = AsdlEF(model, 'classification')
    loss, kron = backend.kron(xi, yi, N=1)
    kron_true = 7 * kron
    loss_true = 7 * loss
    X = torch.repeat_interleave(xi, 7, 0)
    y = torch.repeat_interleave(yi, 7, 0)
    loss_test, kron_test  = backend.kron(X, y, N=7)
    assert torch.allclose(kron_true.diag(), kron_test.diag())
    assert torch.allclose(loss_true, loss_test)


def test_kron_ggn_reg_asdl_vs_diag_reg(reg_Xy, model):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = reg_Xy
    backend = AsdlGGN(model, 'regression', stochastic=False)
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    assert len(dggn) == model.n_params
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


def test_diag_ggn_hetreg_asdl(het_reg_Xy, het_model):
    X, y = het_reg_Xy
    backend = AsdlGGN(het_model, 'heteroscedastic_regression', stochastic=False)
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == het_model.n_params

    # check against manually computed full GGN:
    backend = AsdlGGN(het_model, 'heteroscedastic_regression', stochastic=False)
    loss_f, H_ggn = backend.full(X, y)
    assert loss == loss_f
    loss, h = backend.kron(X, y, N=len(y))
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_kron_ggn_hetreg_asdl_vs_diag_reg(het_reg_Xy, het_model):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = het_reg_Xy
    backend = AsdlGGN(het_model, 'heteroscedastic_regression', stochastic=False)
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)


def test_diag_ef_hetreg_asdl(het_reg_Xy, het_model):
    X, y = het_reg_Xy
    backend = AsdlEF(het_model, 'heteroscedastic_regression')
    loss, dggn = backend.diag(X, y)
    # sanity check size of diag ggn
    assert len(dggn) == het_model.n_params

    # check against manually computed full GGN:
    backend = AsdlEF(het_model, 'heteroscedastic_regression')
    loss_f, H_ggn = backend.full(X, y)
    assert loss == loss_f
    loss, h = backend.kron(X, y, N=len(y))
    assert torch.allclose(dggn, H_ggn.diagonal())


def test_kron_ef_hetreg_asdl_vs_diag_reg(het_reg_Xy, het_model):
    # For a single data point, Kron is exact and should equal diag GGN
    X, y = het_reg_Xy
    backend = AsdlGGN(het_model, 'heteroscedastic_regression')
    loss, dggn = backend.diag(X[:1], y[:1], N=1)
    # sanity check size of diag ggn
    loss, kron = backend.kron(X[:1], y[:1], N=1)
    assert torch.allclose(kron.diag(), dggn)
