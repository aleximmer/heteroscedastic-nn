from functools import partial
import logging
import numpy as np
import torch
from torch.nn.utils.convert_parameters import vector_to_parameters
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.nn import CrossEntropyLoss, MSELoss
from torch.nn.utils import parameters_to_vector
from torch.distributions import Normal
import wandb

from hetreg.utils import wandb_log_prior, wandb_log_parameter_norm

GB_FACTOR = 1024 ** 3

from hetreg.betanll_utils import beta_nll_loss, gaussian_log_likelihood_loss, faithful_loss, gaussian_beta_log_likelihood_loss
from asdfghjkl.loss import heteroscedastic_mse_loss

from bayesian_torch.models.dnn_to_bnn import get_kl_loss, dnn_to_bnn

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()



def expand_prior_precision(prior_prec, model):
    theta = parameters_to_vector(model.parameters())
    device, P = theta.device, len(theta)
    assert prior_prec.ndim == 1
    if len(prior_prec) == 1:  # scalar
        return torch.ones(P, device=device) * prior_prec
    elif len(prior_prec) == P:  # full diagonal
        return prior_prec.to(device)
    else:
        return torch.cat([delta * torch.ones_like(m).flatten() for delta, m
                          in zip(prior_prec, model.parameters())])


def get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device):
    log_prior_prec_init = np.log(prior_prec_init)
    if prior_structure == 'scalar':
        log_prior_prec = log_prior_prec_init * torch.ones(1, device=device)
    elif prior_structure == 'layerwise':
        log_prior_prec = log_prior_prec_init * torch.ones(H, device=device)
    elif prior_structure == 'diagonal':
        log_prior_prec = log_prior_prec_init * torch.ones(P, device=device)
    else:
        raise ValueError(f'Invalid prior structure {prior_structure}')
    log_prior_prec.requires_grad = True
    return log_prior_prec


def valid_performance(model, test_loader, criterion, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            f = model(X)
        perf +=  (f[:, 0] - y.squeeze()).square().sum() / N
        nll += criterion(f, y).mean()
    return perf.item(), nll.item()

def valid_performance_vi(model, test_loader, criterion, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            f_msamples = torch.stack([model(X) for k in range(10)], dim=1)
            mu = f_msamples[:, :, 0].mean(1)
            std = f_msamples[:, :, 1].mean(1)
            f = torch.stack([mu, std], dim=-1)
        perf +=  (f[:, 0] - y.squeeze()).square().sum() / N
        nll += criterion(f, y).mean()
    return perf.item(), nll.item()


def valid_performance_mcdropout(model, test_loader, criterion, device):
    N = len(test_loader.dataset)
    perf = 0
    nll = 0
    for X, y in test_loader:
        X, y = X.detach().to(device), y.detach().to(device)
        with torch.no_grad():
            model.eval()
            enable_dropout(model)
            f_msamples = torch.stack([model(X) for k in range(10)], dim=1)
            mu = f_msamples[:, :, 0].mean(1)
            var = f_msamples[:, :, 1].mean(1)
            f = torch.stack([mu, var], dim=-1)
        perf +=  (f[:, 0] - y.squeeze()).square().sum() / N
        nll += criterion(f, y).mean()
    return perf.item(), nll.item()


def get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min):
    n_steps = n_epochs * len(train_loader)
    if scheduler == 'exp':
        min_lr_factor = lr_min / lr
        gamma = np.exp(np.log(min_lr_factor) / n_steps)
        return ExponentialLR(optimizer, gamma=gamma)
    elif scheduler == 'cos':
        return CosineAnnealingLR(optimizer, n_steps, eta_min=lr_min)
    else:
        raise ValueError(f'Invalid scheduler {scheduler}')


def get_model_optimizer(optimizer, model, lr, weight_decay=0):
    if optimizer == 'Adam':
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        # fixup parameters should have 10x smaller learning rate
        is_fixup = lambda param: param.size() == torch.Size([1])  # scalars
        fixup_params = [p for p in model.parameters() if is_fixup(p)]
        standard_params = [p for p in model.parameters() if not is_fixup(p)]
        params = [{'params': standard_params}, {'params': fixup_params, 'lr': lr / 10.}]
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f'Invalid optimizer {optimizer}')


def gradient_to_vector(parameters):
    return parameters_to_vector([e.grad for e in parameters])


def vector_to_gradient(vec, parameters):
    return vector_to_parameters(vec, [e.grad for e in parameters])


def betalik_optimization(model,
                         train_loader,
                         valid_loader=None,
                         n_epochs=500,
                         beta=0.5,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='cos',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         temperature=1.,
                         use_wandb=False):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    n_epochs : int
    beta: float
        beta value for beta-NLL loss
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    temperature : float

    Returns
    -------
    model : torch.nn.Module
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)

    if use_wandb:
        wandb.config.update(dict(n_data=N), allow_val_change=True)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)


    losses = list()
    valid_perfs = list()
    valid_nlls = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)
        criterion = gaussian_log_likelihood_loss
        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()

            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)
            theta = parameters_to_vector(model.parameters())
            crit_factor = 1 / temperature

            f = model(X)


            loss = gaussian_beta_log_likelihood_loss(f[:,0].unsqueeze(1), f[:,1].unsqueeze(1), target=y, beta=beta).mean() + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).mean().item()
            epoch_perf += (f[:,0].detach() - y.squeeze()).square().sum() / N


        losses.append(epoch_loss)
        #logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                val_criterion = criterion
                val_perf, val_nll = valid_performance(model, valid_loader, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                #logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        if use_wandb:
            wandb.log(epoch_log, step=epoch, commit=(epoch % 50) == 0)
        if use_wandb and (epoch % 50) == 0:
            wandb_log_parameter_norm(model)

    return model, valid_perfs, valid_nlls


def mcdropout_optimization(model,
                         train_loader,
                         valid_loader=None,
                         n_epochs=500,
                         beta=0.5,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='cos',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         temperature=1.,
                         use_wandb=False):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    n_epochs : int
    beta: float
        beta value for beta-NLL loss
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    temperature : float

    Returns
    -------
    model : torch.nn.Module
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)

    if use_wandb:
        wandb.config.update(dict(n_data=N), allow_val_change=True)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)


    losses = list()
    valid_perfs = list()
    valid_nlls = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)
        criterion = gaussian_log_likelihood_loss
        # standard NN training per batch
        torch.cuda.empty_cache()
        model.train()
        for c,(X, y) in enumerate(train_loader):
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()

            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)
            theta = parameters_to_vector(model.parameters())
            crit_factor = 1 / temperature

            f = model(X)


            loss = gaussian_log_likelihood_loss(f, target=y).mean() + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).mean().item()
            epoch_perf += (f[:,0].detach() - y.squeeze()).square().sum() / N


        losses.append(epoch_loss)
        #logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        # compute validation error to report during training
        if valid_loader is not None:
            model.eval()
            with torch.no_grad():
                val_criterion = criterion
                val_perf, val_nll = valid_performance(model, valid_loader, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                #logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        if use_wandb:
            wandb.log(epoch_log, step=epoch, commit=(epoch % 50) == 0)
        if use_wandb and (epoch % 50) == 0:
            wandb_log_parameter_norm(model)

    return model, valid_perfs, valid_nlls

def vi_optimization(model,
                         train_loader,
                         valid_loader=None,
                         n_epochs=500,
                         beta=0.5,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='cos',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         temperature=1.,
                         use_wandb=False,
                         double=True):
                         #vi_prior_mu=0.0,
                         #vi_posterior_mu_init=0.0,
                         #vi_posterior_rho_init=-3.0):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    n_epochs : int
    beta: float
        beta value for beta-NLL loss
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    temperature : float

    Returns
    -------
    model : torch.nn.Module
    losses : list
    """
    # if lr_min is None:  # don't decay lr
    #     lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    # H = len(list(model.parameters()))
    # P = len(parameters_to_vector(model.parameters()))
    # log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)
    # prior_prec = torch.exp(log_prior_prec).detach()
    #
    if use_wandb:
        wandb.config.update(dict(n_data=N), allow_val_change=True)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)


    losses = list()
    valid_perfs = list()
    valid_nlls = list()

    if double:
        model = model.double()

    for epoch in range(1, n_epochs + 1):
        # if epoch % 100 == 0:
        #     print('Epoch {}'.format(epoch))
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)
        criterion = gaussian_log_likelihood_loss
        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()

            f = model(X)

            kl = get_kl_loss(model)

            loss = gaussian_log_likelihood_loss(f, target=y).mean() + (kl/N) #+ (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            # with torch.no_grad():
            #     f = model(X)
            epoch_nll += criterion(f.detach(), y).mean().item()
            epoch_perf += (f[:,0].detach() - y.squeeze()).square().sum() / N
            # print(epoch_nll)

        losses.append(epoch_loss)
        #logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                val_criterion = criterion
                val_perf, val_nll = valid_performance_vi(model, valid_loader, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                #logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        if use_wandb:
            wandb.log(epoch_log, step=epoch, commit=(epoch % 50) == 0)
        if use_wandb and (epoch % 50) == 0:
            wandb_log_parameter_norm(model)

    return model, valid_perfs, valid_nlls


def faithful_optimization(model,
                         train_loader,
                         valid_loader=None,
                         n_epochs=500,
                         beta=0.5,
                         lr=1e-3,
                         lr_min=None,
                         optimizer='Adam',
                         scheduler='cos',
                         prior_structure='layerwise',
                         prior_prec_init=1.,
                         temperature=1.,
                         use_wandb=False):
    """Runs marglik optimization training for a given model and training dataloader.

    Parameters
    ----------
    model : torch.nn.Module
        torch model
    train_loader : DataLoader
        pytorch training dataset loader
    valid_loader : DataLoader
    n_epochs : int
    beta: float
        beta value for beta-NLL loss
    lr : float
        learning rate for model optimizer
    lr_min : float
        minimum learning rate, defaults to lr and hence no decay
        to have the learning rate decay from 1e-3 to 1e-6, set
        lr=1e-3 and lr_min=1e-6.
    optimizer : str
        either 'Adam' or 'SGD'
    scheduler : str
        either 'exp' for exponential and 'cos' for cosine decay towards lr_min
    prior_structure : str
        'scalar', 'layerwise', 'diagonal'
    prior_prec_init : float
        initial prior precision
    temperature : float

    Returns
    -------
    model : torch.nn.Module
    losses : list
    """
    if lr_min is None:  # don't decay lr
        lr_min = lr
    device = parameters_to_vector(model.parameters()).device
    N = len(train_loader.dataset)
    H = len(list(model.parameters()))
    P = len(parameters_to_vector(model.parameters()))
    log_prior_prec = get_prior_hyperparams(prior_prec_init, prior_structure, H, P, device)

    if use_wandb:
        wandb.config.update(dict(n_data=N), allow_val_change=True)

    # set up model optimizer and scheduler
    optimizer = get_model_optimizer(optimizer, model, lr)
    scheduler = get_scheduler(scheduler, optimizer, train_loader, n_epochs, lr, lr_min)


    losses = list()
    valid_perfs = list()
    valid_nlls = list()

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0
        epoch_perf = 0
        epoch_nll = 0
        epoch_log = dict(epoch=epoch)
        criterion = gaussian_log_likelihood_loss
        # standard NN training per batch
        torch.cuda.empty_cache()
        for X, y in train_loader:
            X, y = X.detach().to(device), y.to(device)
            optimizer.zero_grad()

            prior_prec = torch.exp(log_prior_prec).detach()
            delta = expand_prior_precision(prior_prec, model)
            theta = parameters_to_vector(model.parameters())
            crit_factor = 1 / temperature

            f = model(X)


            loss = faithful_loss(f[:,0].unsqueeze(1), f[:,1].unsqueeze(1), target=y).mean() + (0.5 * (delta * theta) @ theta) / N / crit_factor
            loss.backward()
            optimizer.step()

            epoch_loss += loss.cpu().item() / len(train_loader)
            epoch_nll += criterion(f.detach(), y).mean().item()
            epoch_perf += (f[:,0].detach() - y.squeeze()).square().sum() / N


        losses.append(epoch_loss)
        #logging.info(f'MARGLIK[epoch={epoch}]: train. perf={epoch_perf:.2f}; loss={epoch_loss:.5f}; nll={epoch_nll:.5f}')
        optimizer.zero_grad(set_to_none=True)
        llr = scheduler.get_last_lr()[0]
        epoch_log.update({'train/loss': epoch_loss, 'train/nll': epoch_nll, 'train/perf': epoch_perf, 'train/lr': llr})
        # compute validation error to report during training
        if valid_loader is not None:
            with torch.no_grad():
                val_criterion = criterion
                val_perf, val_nll = valid_performance(model, valid_loader, val_criterion, device)
                valid_perfs.append(val_perf)
                valid_nlls.append(val_nll)
                #logging.info(f'MARGLIK[epoch={epoch}]: valid. perf={val_perf:.2f}; nll={val_nll:.5f}.')
                epoch_log.update({'valid/perf': val_perf, 'valid/nll': val_nll})

        if use_wandb:
            wandb.log(epoch_log, step=epoch, commit=(epoch % 50) == 0)
        if use_wandb and (epoch % 50) == 0:
            wandb_log_parameter_norm(model)

    return model, valid_perfs, valid_nlls
