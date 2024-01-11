from math import sqrt, log
import logging
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from torch.distributions import Normal
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

from laplace.curvature.asdl import AsdlGGN, AsdlEF

from hetreg.marglik import marglik_optimization
from hetreg.utils import TensorDataLoader, set_seed, get_laplace_approximation
from hetreg.uci_datasets import UCI_DATASETS, UCIRegressionDatasets
from hetreg.models import MLP, ACTIVATIONS, HEADS, NaturalReparamHead

# Beta-NLL imports
from hetreg.betanll import betalik_optimization, faithful_optimization, vi_optimization, enable_dropout, mcdropout_optimization
# Faithful imports
from hetreg.models import MLPFaithful

from hetreg.crispr_datasets import CrisprDatasets, CRISPR_DATASETS

from hetreg.models import make_bayesian

def main(seed, dataset, width, depth, activation, head, lr, lr_min, n_epochs, batch_size, method, beta, likelihood,
         prior_prec_init, approx, lr_hyp, lr_hyp_min, n_epochs_burnin, marglik_frequency, n_hypersteps,
         device, data_root, use_wandb, optimizer, head_activation, double, marglik_early_stopping, vi_prior_mu,
          vi_posterior_mu_init, vi_posterior_rho_init, typeofrep, rep):
    set_seed(seed)

    # Set up data. 90% train and 10% test is default since Hernandez-Lobato
    # Take 10% of train for validation when not using marglik
    ds_kwargs = dict(
        split_train_size=0.9, split_valid_size=0.1, root=data_root, seed=seed, double=double
    )
    if dataset in UCI_DATASETS:
        ds_train = UCIRegressionDatasets(dataset, split='train', **ds_kwargs)
        ds_valid = UCIRegressionDatasets(dataset, split='valid', **ds_kwargs)
        ds_train_full = UCIRegressionDatasets(dataset, split='train', **{**ds_kwargs, **{'split_valid_size': 0.0}})
        ds_test = UCIRegressionDatasets(dataset, split='test', **ds_kwargs)
        assert len(ds_train) + len(ds_valid) == len(ds_train_full)
    else:
        ds_train = CrisprDatasets(dataset, split='train', rep = rep, **ds_kwargs)
        ds_valid = CrisprDatasets(dataset, split='valid', rep = rep, **ds_kwargs)
        ds_train_full = CrisprDatasets(dataset, split='train', rep = rep, **{**ds_kwargs, **{'split_valid_size': 0.0}})
        ds_test = CrisprDatasets(dataset, split='test', rep = rep, **ds_kwargs)
        assert len(ds_train) + len(ds_valid) == len(ds_train_full)
    train_loader = TensorDataLoader(ds_train.data.to(device), ds_train.targets.to(device), batch_size=batch_size)
    valid_loader = TensorDataLoader(ds_valid.data.to(device), ds_valid.targets.to(device), batch_size=batch_size)
    train_loader_full = TensorDataLoader(ds_train_full.data.to(device), ds_train_full.targets.to(device), batch_size=batch_size)
    test_loader = TensorDataLoader(ds_test.data.to(device), ds_test.targets.to(device), batch_size=batch_size)

    # Set up model.
    input_size = ds_train.data.size(1)

    test_loglik_bayes = None
    if method in ['map', 'marglik']:
        output_size = 1 if likelihood == 'homoscedastic' else 2
        head = None if likelihood == 'homoscedastic' else head
        if head == 'meanvar':
            # decompose into mean-var and then reparam into natural for marglik
            head = 'gaussian'
            mean_head = NaturalReparamHead
        else:
            mean_head = None
        model = MLP(
            input_size, width, depth, output_size=output_size, activation=activation,
            head=head, head_activation=head_activation
        ).to(device)
        if double:
            model = model.double()

        # Train model.
        # backend = AsdlGGN if curv == 'ggn' else AsdlEF
        backend = AsdlGGN
        laplace = get_laplace_approximation(approx)
        lh = 'heteroscedastic_regression' if likelihood == 'heteroscedastic' else 'regression'
        if method == 'marglik':
            la, model, _, _, _ = marglik_optimization(
                model, train_loader_full, likelihood=lh, lr=lr, lr_min=lr_min, lr_hyp=lr_hyp, early_stopping=marglik_early_stopping,
                lr_hyp_min=lr_hyp_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency,
                laplace=laplace, prior_structure='layerwise', backend=backend, n_epochs_burnin=n_epochs_burnin,
                scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec_init, use_wandb=use_wandb, mean_head=mean_head
            )
        else:  # MAP with CV
            # evaluate prior precisions on the validation set
            prior_precs = np.logspace(-4, 4, 9)
            nlls = []
            for prior_prec in prior_precs:
                print(prior_prec)
                model.reset_parameters()
                la, model, margliks, valid_perfs, valid_nlls = marglik_optimization(
                    model, train_loader, valid_loader=valid_loader, likelihood=lh, lr=lr, lr_min=lr_min, n_epochs=n_epochs, 
                    laplace=laplace, prior_structure='scalar', backend=backend, n_epochs_burnin=n_epochs_burnin,
                    scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=False, mean_head=mean_head
                )
                nlls.append(valid_nlls[-1])

            # choose best prior precision and rerun on combined train + validation set
            opt_prior_prec = prior_precs[np.argmin(nlls)]
            if use_wandb:
                wandb.run.summary['prior_prec_opt'] = opt_prior_prec
                wandb.run.summary['valid/nll'] = np.min(nlls)
            logging.info(f'Best prior precision found: {opt_prior_prec}')
            model.reset_parameters()
            la, model, _, _, _ = marglik_optimization(
                model, train_loader_full, likelihood=lh, lr=lr, lr_min=lr_min, n_epochs=n_epochs, 
                laplace=laplace, prior_structure='scalar', backend=backend, n_epochs_burnin=n_epochs_burnin,
                scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb,
                mean_head=mean_head
            )
            if likelihood == 'homoscedastic':  # need to find observation noise maximum lik
                ssqe = 0
                for x, y in train_loader_full:
                    with torch.no_grad():
                        ssqe += (y - model(x)).square().sum().item() / len(ds_train_full)
                la.sigma_noise = sqrt(ssqe)
                wandb.log({'hyperparams/sigma_noise': sqrt(ssqe)})

        # Evaluate the trained model on test set.
        scale = ds_train.s
        test_mse = 0
        test_loglik = 0
        test_loglik_bayes = 0
        test_loglik_bayes_mean = 0
        test_loglik_bayes_mc = 0
        test_loglik_bayes_mc_lse = 0
        N = len(test_loader.dataset)
        for x, y in test_loader:
            if likelihood == 'homoscedastic':
                f_mu, f_var = la(x)
                f_mu, f_var = f_mu.squeeze(), f_var.squeeze()
                test_mse += (f_mu - y.squeeze()).square().sum() / N
                pred_dist = Normal(loc=f_mu * scale, scale=la.sigma_noise * scale)
                test_loglik += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                y_std = torch.sqrt(f_var + la.sigma_noise.item() ** 2)
                pred_dist = Normal(loc=f_mu * scale, scale=y_std * scale)
                test_loglik_bayes += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
            elif likelihood == 'heteroscedastic':
                f_mu, f_var, y_var = la(x, het_approx='natural')
                test_mse += (y.squeeze() - f_mu).square().sum() / N
                pred_dist = Normal(loc=f_mu * scale, scale=torch.sqrt(y_var) * scale)
                test_loglik += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                y_std = torch.sqrt(f_var + y_var)
                pred_dist = Normal(loc=f_mu * scale, scale=y_std * scale)
                test_loglik_bayes += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                f_mu, f_var, y_var = la(x, het_approx='mean')
                y_std = torch.sqrt(f_var + y_var)
                pred_dist = Normal(loc=f_mu * scale, scale=y_std * scale)
                test_loglik_bayes_mean += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                f_mu, fy_var = la(x, het_approx='mc', n_samples=1000)
                pred_dist = Normal(loc=f_mu * scale, scale=torch.sqrt(fy_var) * scale)
                test_loglik_bayes_mc += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                samples = la.predictive_samples(x, n_samples=1000)  # (n_samples, n_test, 2)
                mu_samples = - samples[:, :, 0] / (2 * samples[:, :, 1])
                var_samples = - 0.5 / samples[:, :, 1]
                std_samples = torch.nan_to_num(torch.sqrt(var_samples), nan=1e-9)
                dists = Normal(loc=mu_samples * scale, scale=std_samples * scale)
                log_probs = dists.log_prob(y.reshape(1, -1) * scale)
                S = log_probs.shape[0]
                test_loglik_bayes_mc_lse += (torch.logsumexp(log_probs, dim=0) - log(S)).sum().item() / N
            else:
                raise ValueError('Invalid likelihood')
    elif method == 'betanll':
        #output_size = 1 if likelihood == 'homoscedastic' else 2
        """
        Code to get exact settings as original paper
        head = lambda inp, outp: GaussianLikelihoodHead(
                inp, outp, initial_var=1, max_var=100
        )  # Check other heads

        model = MLP_betanll(
            inp_dim=input_size,
            outp_dim=1, # number of covariates, always fixed to 1 in our experiments
            hidden_dims=[50], # should be right but check
            hidden_activation=torch.nn.ReLU,    #check
            weight_init=torch.nn.init.xavier_uniform, # check
            bias_init=torch.nn.init.zeros_,
            outp_layer=head,
        )

        if double:
            model = model.double()
        """
        output_size = 2
        model = MLP(
            input_size, width, depth, output_size=output_size, activation=activation,
            head=head, head_activation=head_activation
        ).to(device)
        if double:
            model = model.double()

        #grid_setting_names = ['lr', 'prior_precision']
        prior_precs = np.logspace(-4, 4, 9)
        #grid_settings = [[1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 7e-3], prior_precs]
        #for settings in itertools.product(*grid_settings):
        #    conf = {grid_setting_names[idx]: value for idx, value in enumerate(settings)}
        nlls = []
        for prior_prec in prior_precs:
            print(prior_prec)
            model.reset_parameters()
            model, valid_perfs, valid_nlls = betalik_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=beta,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=use_wandb
            )
            nlls.append(valid_nlls[-1])

        # choose best prior precision and rerun on combined train + validation set
        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        model.reset_parameters()
        model, _, _ = betalik_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=beta,
            scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb
        )

        # 0. create model compliant with their restrictions : DONE

        # 1. cross-validate prior precision/weight decay as above: DONE
        # 2. retrain on train_loader_full: DONE
        # 3. compute test performance as above with Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
        #pass
        # Evaluate the trained model on test set.
        scale = ds_train.s
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        for x, y in test_loader:
            f = model(x)
            mu = f[: , 0]
            #std = f[:,1]
            std = torch.sqrt(f[: , 1])
            #test_loglik += -gaussian_log_likelihood_loss(f.detach(), y).sum().item()
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            #Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N


    elif method == 'mcdropout':
        output_size = 2
        prior_precs = np.logspace(-4, 4, 9)
        nlls = []
        for prior_prec in prior_precs:
            print(prior_prec)
            model = MLP(
                input_size, width, depth, output_size=output_size, activation=activation,
                head=head, head_activation=head_activation, dropout=0.05
            ).to(device)
            model.reset_parameters()
            if double:
                model = model.double()
            # make_bayesian(model, prior_mu=vi_prior_mu, prior_sigma=1./prior_prec, posterior_mu_init=vi_posterior_mu_init, posterior_rho_init=vi_posterior_rho_init, typeofrep=typeofrep)
            # print(model)
            model, valid_perfs, valid_nlls = mcdropout_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=0.0,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=use_wandb)  # Beta 0.0 to have NLL
            nlls.append(valid_nlls[-1])

        # choose best prior precision and rerun on combined train + validation set
        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        model = MLP(
            input_size, width, depth, output_size=output_size, activation=activation,
            head=head, head_activation=head_activation, dropout=0.05
        ).to(device)
        model.reset_parameters()
        if double:
            model = model.double()
        # make_bayesian(model, prior_mu=vi_prior_mu, prior_sigma=1./opt_prior_prec, posterior_mu_init=vi_posterior_mu_init, posterior_rho_init=vi_posterior_rho_init, typeofrep=typeofrep)
        model, _, _ = mcdropout_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=0.0,
            scheduler='cos', optimizer=optimizer, use_wandb=use_wandb)

        # 0. create model compliant with their restrictions : DONE

        # 1. cross-validate prior precision/weight decay as above: DONE
        # 2. retrain on train_loader_full: DONE
        # 3. compute test performance as above with Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
        #pass
        # Evaluate the trained model on test set.
        scale = ds_train.s
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        model.eval()
        enable_dropout(model)
        for i, (x, y) in enumerate(test_loader):
            # f = model(x)
            # mu = f[:, 0]
            # std = f[:,1]
            f_msamples = torch.stack([model(x) for k in range(10)], dim=1)
            mu = f_msamples[:, :, 0].mean(1)
            std = torch.sqrt(f_msamples[:, :, 1].mean(1))
            # test_loglik += -gaussian_log_likelihood_loss(f.detach(), y).sum().item()
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            # Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N



    elif method == 'vi':
        #vi_params = {
        #    "vi_prior_mu" : vi_prior_mu,
        #    "vi_posterior_mu_init" : vi_posterior_mu_init,
        #    "vi_posterior_rho_init" : vi_posterior_rho_init
        #}
        #output_size = 1 if likelihood == 'homoscedastic' else 2
        """
        Code to get exact settings as original paper
        head = lambda inp, outp: GaussianLikelihoodHead(
                inp, outp, initial_var=1, max_var=100
        )  # Check other heads

        model = MLP_betanll(
            inp_dim=input_size,
            outp_dim=1, # number of covariates, always fixed to 1 in our experiments
            hidden_dims=[50], # should be right but check
            hidden_activation=torch.nn.ReLU,    #check
            weight_init=torch.nn.init.xavier_uniform, # check
            bias_init=torch.nn.init.zeros_,
            outp_layer=head,
        )

        if double:
            model = model.double()
        """
        output_size = 2

        #grid_setting_names = ['lr', 'prior_precision']
        prior_precs = np.logspace(-4, 4, 9)
        # grid_setting_names = ['priorsigma1', 'priorsigma2']
        # grid_settings = [prior_precs, prior_precs]
        nlls = []
        # confs = []
        for prior_prec in prior_precs:
        # for settings in itertools.product(*grid_settings):
            # conf = {grid_setting_names[idx]: 1./value for idx, value in enumerate(settings)}
            # confs.append(conf)

            #for prior_prec in [1.0]:#prior_precs:
            #print(prior_prec)
            model = MLP(
                input_size, width, depth, output_size=output_size, activation=activation,
                head=head, head_activation=head_activation
            ).to(device)
            model.reset_parameters()
            if double:
                model = model.double()
            make_bayesian(model, prior_mu=vi_prior_mu, prior_sigma=1./prior_prec, posterior_mu_init=vi_posterior_mu_init, posterior_rho_init=vi_posterior_rho_init, typeofrep=typeofrep)
            # print(model)
            model, valid_perfs, valid_nlls = vi_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=0.0,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, use_wandb=use_wandb , double=double)  # Beta 0.0 to have NLL
            nlls.append(valid_nlls[-1])

        # choose best prior precision and rerun on combined train + validation set
        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        model = MLP(
            input_size, width, depth, output_size=output_size, activation=activation,
            head=head, head_activation=head_activation
        ).to(device)
        model.reset_parameters()
        if double:
            model = model.double()
        make_bayesian(model, prior_mu=vi_prior_mu, prior_sigma=1./opt_prior_prec, posterior_mu_init=vi_posterior_mu_init, posterior_rho_init=vi_posterior_rho_init, typeofrep=typeofrep)
        model, _, _ = vi_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=0.0,
            scheduler='cos', optimizer=optimizer, use_wandb=use_wandb, double=double)

        # 0. create model compliant with their restrictions : DONE

        # 1. cross-validate prior precision/weight decay as above: DONE
        # 2. retrain on train_loader_full: DONE
        # 3. compute test performance as above with Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
        #pass
        # Evaluate the trained model on test set.
        scale = ds_train.s
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        for x, y in test_loader:
            #f = model(x)
            #mu = f[:, 0]
            #std = f[:,1]
            f_msamples = torch.stack([model(x) for k in range(10)], dim=1)
            mu = f_msamples[:, :, 0].mean(1)
            std = torch.sqrt(f_msamples[:, :, 1].mean(1))
            #test_loglik += -gaussian_log_likelihood_loss(f.detach(), y).sum().item()
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            #Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N

    elif method == 'faithful':
        output_size = 2
        model = MLPFaithful(
            input_size, width, depth, activation=activation,
            head=head, head_activation=head_activation
        ).to(device)
        if double:
            model = model.double()

        #grid_setting_names = ['lr', 'prior_precision']
        prior_precs = np.logspace(-4, 4, 9)
        #grid_settings = [[1e-4, 3e-4, 7e-4, 1e-3, 3e-3, 7e-3], prior_precs]
        #for settings in itertools.product(*grid_settings):
        #    conf = {grid_setting_names[idx]: value for idx, value in enumerate(settings)}
        nlls = []
        for prior_prec in prior_precs:
            print(prior_prec)
            model.reset_parameters()
            model, valid_perfs, valid_nlls = faithful_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=beta,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=use_wandb
            )
            nlls.append(valid_nlls[-1])

        # choose best prior precision and rerun on combined train + validation set
        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        model.reset_parameters()
        model, _, _ = faithful_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=beta,
            scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb
        )

        # 0. create model compliant with their restrictions : DONE

        # 1. cross-validate prior precision/weight decay as above: DONE
        # 2. retrain on train_loader_full: DONE
        # 3. compute test performance as above with Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
        #pass
        # Evaluate the trained model on test set.
        scale = ds_train.s
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        for x, y in test_loader:
            f = model(x)
            mu = f[: , 0]
            #std = f[:,1]
            std = torch.sqrt(f[: , 1])
            #test_loglik += -gaussian_log_likelihood_loss(f.detach(), y).sum().item()
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            #Normal(scale * mu, scale * std).log_prob(y).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N

    else:
        raise ValueError('Invalid method')

    if use_wandb:
        wandb.run.summary['test/mse'] = test_mse
        wandb.run.summary['test/loglik'] = test_loglik
        if method in ['map', 'marglik']:
            wandb.run.summary['test/loglik_bayes'] = test_loglik_bayes
            if likelihood == 'heteroscedastic':
                wandb.run.summary['test/loglik_bayes_mean'] = test_loglik_bayes_mean
                wandb.run.summary['test/loglik_bayes_mc'] = test_loglik_bayes_mc
                wandb.run.summary['test/loglik_bayes_mc_lse'] = test_loglik_bayes_mc_lse
    if test_loglik_bayes is not None:
        logging.info(f'Final test performance: MSE={test_mse:.3f}, LogLik={test_loglik:.3f}, LogLikBayes={test_loglik_bayes:.3f}')
    else:
        logging.info(f'Final test performance: MSE={test_mse:.3f}, LogLik={test_loglik:.3f}')


if __name__ == '__main__':
    import sys
    import argparse
    from arg_utils import set_defaults_with_yaml_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=7, type=int)
    parser.add_argument('--dataset', default='boston-housing', choices=UCI_DATASETS+CRISPR_DATASETS)
    # architecture
    parser.add_argument('--width', default=50, type=int)
    parser.add_argument('--depth', default=2, type=int)
    parser.add_argument('--activation', default='relu', choices=ACTIVATIONS)
    parser.add_argument('--head', default='natural', choices=HEADS)
    parser.add_argument('--head_activation', default='softplus', choices=['exp', 'softplus'])
    # optimization (general)
    parser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'])
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--lr_min', default=1e-3, type=float, help='Cosine decay target')
    parser.add_argument('--n_epochs', default=500, type=int)
    parser.add_argument('--batch_size', default=-1, type=int)
    parser.add_argument('--method', default='faithful', help='Method', choices=['map', 'marglik', 'betanll', 'faithful', 'vi', 'mcdropout'])
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--likelihood', default='heteroscedastic', choices=['heteroscedastic', 'homoscedastic'])
    parser.add_argument('--prior_prec_init', default=1.0, type=float, help='Prior precision init or final for MAP.')
    # marglik-specific
    parser.add_argument('--approx', default='full', choices=['full', 'kron', 'diag', 'kernel'])
    # parser.add_argument('--curv', default='ggn', choices=['ggn', 'ef'])
    parser.add_argument('--lr_hyp', default=0.1, type=float)
    parser.add_argument('--lr_hyp_min', default=0.1, type=float)
    parser.add_argument('--n_epochs_burnin', default=10, type=int)
    parser.add_argument('--marglik_frequency', default=50, type=int)
    parser.add_argument('--n_hypersteps', default=50, help='Number of steps on every marglik estimate (partial grad accumulation)', type=int)
    parser.add_argument('--marglik_early_stopping', default=False, action=argparse.BooleanOptionalAction)
    # others
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--use_wandb', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--double', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--config', nargs='+')
    # parser.add_argument('--wandbdirspec', type=str, default='')
    parser.add_argument('--vi-prior-mu', default=0.0, type=float)
    parser.add_argument('--vi-posterior-mu-init', default=0.0, type=float)
    parser.add_argument('--vi-posterior-rho-init', default=-3.0, type=float)
    parser.add_argument('--typeofrep', default="Flipout", choices=['Flipout', 'Reparameterization'])
    parser.add_argument('--rep', default=0, type=int, help="Replicate to use as response for CRISPR experiments."
                                                           "0 for averaged replicates as response."
                                                           " DOES ONLY MAKE SENSE FOR EXPERIMENTS ON CRISPR DATASETS!")

    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    args.pop('config')
    if args['method'] == 'map':
        # do not do marglik optimization
        args['n_epochs_burnin'] = args['n_epochs'] + 1
    if args['method'] == 'betanll':
        args['head'] = 'gaussian'
    if args['method'] == 'faithful':
        args['head'] = 'gaussian'
    if args['method'] in ['vi', 'mcdropout']:
        args['head'] = 'gaussian'
        args['beta'] = 0.0
    if args['use_wandb']:
        import uuid
        import copy
        tags = [args['dataset'], args['method']]
        config = copy.deepcopy(args)
        run_name = '-'.join(tags)
        run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
        load_dotenv()
        wandb.init(project='uci-experiments',
                   mode='online', config=config, name=run_name, tags=tags)
                   # dir=args['wandbdirspec'])
    # args.pop('wandbdirspec')
    print(args)
    main(**args)
