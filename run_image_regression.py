from math import sqrt, log
import logging
import torch
import wandb
import numpy as np
from dotenv import load_dotenv
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
logging.basicConfig(format='[%(filename)s:%(lineno)s]%(levelname)s: %(message)s', level=logging.INFO)

from laplace.curvature.asdl import AsdlGGN

from hetreg.marglik import marglik_optimization
from hetreg.utils import TensorDataLoader, set_seed, get_laplace_approximation, dataset_to_tensors, Timer
from hetreg.image_datasets import IMAGE_DATASETS, get_dataset
from hetreg.models import MLP, ACTIVATIONS, HEADS, LeNet, ResNet, ResNetFaithful, MLPFaithful, LeNetFaithful, NaturalReparamHead, make_bayesian, LeNetMCDrpt
from hetreg.betanll import betalik_optimization, faithful_optimization, vi_optimization, enable_dropout, mcdropout_optimization


def main(seed, model, dataset, head, lr, lr_min, n_epochs, batch_size, method, beta, likelihood,
         prior_prec_init, approx, lr_hyp, lr_hyp_min, n_epochs_burnin, marglik_frequency, n_hypersteps,
         device, data_root, use_wandb, head_activation, marglik_early_stopping, vi_prior_mu, download_data,
         vi_posterior_mu_init, vi_posterior_rho_init, typeofrep, optimizer, activation, double, n_epochs_val, het_noise):
    set_seed(seed)
    ds_train, ds_test = get_dataset(
        dataset, data_root, seed, het_noise=het_noise, download=download_data
    )
    if 'mnist' in dataset:
        channels, pixels = 1, 28
    else:
        channels, pixels = 3, 32

    x_train, y_train = dataset_to_tensors(ds_train, device=device)
    if double:
        x_train, y_train = x_train.double(), y_train.double()
    train_loader_full = TensorDataLoader(x_train, y_train, batch_size=batch_size, shuffle=True)
    # split x_train and y_train into train and validation set with random 90/10 split
    perm = torch.randperm(len(x_train), device=device)
    split = int(len(x_train) * 0.9)
    train_loader = TensorDataLoader(x_train[perm[:split]], y_train[perm[:split]], batch_size=batch_size, shuffle=True)
    valid_loader = TensorDataLoader(x_train[perm[split:]], y_train[perm[split:]], batch_size=batch_size, shuffle=False)
    x_test, y_test = dataset_to_tensors(ds_test, device=device)
    if double:
        x_test, y_test = x_test.double(), y_test.double()
    test_loader = TensorDataLoader(x_test, y_test, batch_size=batch_size, shuffle=False)

    # Set up model.
    output_size = 1 if likelihood == 'homoscedastic' else 2
    head = None if likelihood == 'homoscedastic' else head
    if head == 'meanvar':
        # decompose into mean-var and then reparam into natural for marglik
        head = 'gaussian'
        mean_head = NaturalReparamHead
    else:
        mean_head = None

    if method not in ['vi', 'mcdropout']:
        if model == 'mlp':
            if method in ['map', 'marglik', 'betanll']:
                model = MLP(
                    input_size=channels * pixels ** 2, width=500, depth=3, output_size=output_size,
                    activation=activation, head=head, head_activation=head_activation, skip_head=head is None
                ).to(device)
            elif method == 'faithful':
                model = MLPFaithful(
                    input_size=channels * pixels ** 2, width=500, depth=3, activation=activation,
                    head=head, head_activation=head_activation
                ).to(device)
        elif model == 'cnn':
            if method in ['map', 'marglik', 'betanll']:
                model = LeNet(
                    in_channels=channels, n_out=output_size, activation=activation, n_pixels=pixels,
                    head=head, head_activation=head_activation, skip_head=head is None
                ).to(device)
            elif method == 'faithful':
                model = LeNetFaithful(
                    in_channels=channels, n_out=output_size, activation=activation, n_pixels=pixels,
                    head=head, head_activation=head_activation
                ).to(device)
        elif model == 'resnet':
            if method in ['map', 'marglik', 'betanll']:
                model = ResNet(depth=20, in_planes=32, in_channels=3, n_out=output_size,
                               head=head, head_activation=head_activation, skip_head=head is None).to(device)
            elif method == 'faithful':
                model = ResNetFaithful(depth=20, in_planes=32, in_channels=3, n_out=output_size,
                                       head=head, head_activation=head_activation).to(device)
        else:
            raise ValueError('Invalid model.')
        set_seed(seed)
        model.reset_parameters()
        if double:
            model = model.double()
    else:
        model_name_for_vi = model


    test_loglik_bayes = None
    if method in ['map', 'marglik']:

        # Train model.
        laplace = get_laplace_approximation(approx)
        lh = 'heteroscedastic_regression' if likelihood == 'heteroscedastic' else 'regression'
        if method == 'marglik':
            la, model, _, _, _ = marglik_optimization(
                model, train_loader_full, likelihood=lh, lr=lr, lr_min=lr_min, lr_hyp=lr_hyp, early_stopping=marglik_early_stopping,
                lr_hyp_min=lr_hyp_min, n_epochs=n_epochs, n_hypersteps=n_hypersteps, marglik_frequency=marglik_frequency,
                laplace=laplace, prior_structure='layerwise', backend=AsdlGGN, n_epochs_burnin=n_epochs_burnin,
                scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec_init, use_wandb=use_wandb, mean_head=mean_head,
                grad_clip_norm=10.0
            )
        else:
            prior_precs = np.logspace(-2, 4, 7)
            nlls = []
            for prior_prec in prior_precs:
                set_seed(seed)
                model.reset_parameters()
                _, model, _, valid_perfs, valid_nlls = marglik_optimization(
                    model, train_loader, valid_loader=valid_loader, likelihood=lh, lr=lr, lr_min=lr_min, n_epochs=n_epochs_val,
                    laplace=laplace, prior_structure='scalar', backend=AsdlGGN, n_epochs_burnin=n_epochs_burnin,
                    scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=False, fit_laplace=False,
                    grad_clip_norm=10.0, mean_head=mean_head
                )
                nlls.append(valid_nlls[-1])
                logging.info(f'Validation performance for prior prec {prior_prec}: {nlls[-1]}')
                if use_wandb:
                    wandb.run.summary[f'valnll/prior_prec_{prior_prec}'] = valid_nlls[-1]

            # choose best prior precision and rerun on combined train + validation set
            opt_prior_prec = prior_precs[np.nanargmin(nlls)]
            if use_wandb:
                wandb.run.summary['prior_prec_opt'] = opt_prior_prec
                wandb.run.summary['valid/nll'] = np.nanmin(nlls)
            logging.info(f'Best prior precision found: {opt_prior_prec}')
            set_seed(seed)
            model.reset_parameters()
            la, model, _, _, _ = marglik_optimization(
                model, train_loader_full, likelihood=lh, lr=lr, lr_min=lr_min, n_epochs=n_epochs,
                laplace=laplace, prior_structure='scalar', backend=AsdlGGN, n_epochs_burnin=n_epochs_burnin,
                scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb,
                fit_laplace=True, grad_clip_norm=10.0, mean_head=mean_head
            )
            if likelihood == 'homoscedastic':  # need to find observation noise maximum lik
                ssqe = 0
                for x, y in train_loader_full:
                    with torch.no_grad():
                        ssqe += (y - model(x)).square().sum().item() / len(ds_train)
                la.sigma_noise = sqrt(ssqe)
                wandb.log({'hyperparams/sigma_noise': sqrt(ssqe)})

        wandb.run.define_metric('predictive', summary='mean')
        wandb.run.define_metric('predictive_natural', summary='mean')
        wandb.run.define_metric('predictive_mean', summary='mean')
        wandb.run.define_metric('predictive_mc', summary='mean')
        scale = ds_train.std_target
        test_mse = 0
        test_loglik = 0
        test_loglik_bayes = 0
        test_loglik_bayes_mean = 0
        test_loglik_bayes_mc = 0
        test_loglik_bayes_mc_lse = 0
        fs = []
        fse = []
        f_mus, f_stds = list(), list()
        reps = list()
        N = len(test_loader.dataset)
        for i, (x, y) in enumerate(test_loader):
            reps.append(model.representation(x).cpu().numpy())
            with Timer('mapred', wandb=True, logger=True, step=i):
                with torch.no_grad():
                    model(x).mean()
            if likelihood == 'homoscedastic':
                with Timer('predictive', wandb=True, logger=True, step=i):
                    f_mu, f_var = la(x)
                f_mu, f_var = f_mu.squeeze(), f_var.squeeze()
                f_mus.append(f_mu.detach().cpu() * scale)
                f_stds.append(torch.ones_like(f_mu).cpu() * la.sigma_noise.item() * scale)
                test_mse += (f_mu - y.squeeze()).square().sum() / N
                pred_dist = Normal(loc=f_mu * scale, scale=la.sigma_noise * scale)
                test_loglik += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                y_std = torch.sqrt(f_var + la.sigma_noise.item() ** 2)
                fse.append(y_std.square().detach().cpu().numpy())
                pred_dist = Normal(loc=f_mu * scale, scale=y_std * scale)
                test_loglik_bayes += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
            elif likelihood == 'heteroscedastic':
                with Timer('predictive_natural', wandb=True, logger=True, step=i):
                    f_mu, f_var, y_var = la(x, het_approx='natural')
                f_mus.append(f_mu.detach().cpu() * scale)
                f_stds.append(torch.sqrt(y_var).cpu() * scale)
                fs.append(y_var.detach().cpu().numpy())
                test_mse += (y.squeeze() - f_mu).square().sum() / N
                pred_dist = Normal(loc=f_mu * scale, scale=torch.nan_to_num(torch.sqrt(y_var), nan=1.0) * scale)
                test_loglik += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                y_std = torch.nan_to_num(torch.sqrt(f_var + y_var), nan=1.0)
                fse.append(y_std.square().detach().cpu().numpy())
                pred_dist = Normal(loc=f_mu * scale, scale=y_std * scale)
                test_loglik_bayes += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                with Timer('predictive_mean', wandb=True, logger=True, step=i):
                    f_mu, f_var, y_var = la(x, het_approx='mean')
                y_std = torch.nan_to_num(torch.sqrt(f_var + y_var), nan=1.0)
                pred_dist = Normal(loc=f_mu * scale, scale=y_std * scale)
                test_loglik_bayes_mean += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                try:
                    fy_mu, fy_var = la(x, het_approx='mc', n_samples=100)
                except:
                    fy_mu, fy_var = f_mu, f_var + y_var
                f_mu = fy_mu
                pred_dist = Normal(loc=f_mu * scale, scale=torch.nan_to_num(torch.sqrt(fy_var), nan=1.0) * scale)
                test_loglik_bayes_mc += pred_dist.log_prob(y.squeeze() * scale).sum().item() / N
                with Timer('predictive_mc', wandb=True, logger=True, step=i):
                    samples = la.predictive_samples(x, n_samples=100)  # (n_samples, n_test, 2)
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
        prior_precs = np.logspace(-2, 4, 7)
        nlls = []
        for prior_prec in prior_precs:
            set_seed(seed)
            model.reset_parameters()
            model, valid_perfs, valid_nlls = betalik_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs_val, beta=beta,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=False
            )
            nlls.append(valid_nlls[-1])
            if use_wandb:
                wandb.run.summary[f'valnll/prior_prec_{prior_prec}'] = valid_nlls[-1]
        opt_prior_prec = prior_precs[np.nanargmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.nanmin(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        set_seed(seed)
        model.reset_parameters()
        model, _, _ = betalik_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=beta,
            scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb
        )

        # evaluate
        wandb.run.define_metric('predictive', summary='mean')
        scale = ds_train.std_target
        test_mse = 0
        test_loglik = 0
        fs = []
        f_mus, f_stds = list(), list()
        reps = []
        N = len(test_loader.dataset)
        for i, (x, y) in enumerate(test_loader):
            with Timer('predictive', wandb=True, logger=True, step=i):
                with torch.no_grad():
                    f = model(x).detach()
            mu = f[: , 0]
            std = torch.sqrt(f[: , 1])
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N
            fs.append(std.detach().cpu().numpy())
            reps.append(model.representation(x).cpu().numpy())
            f_mus.append(mu.detach().cpu() * scale)
            f_stds.append(std.detach().cpu() * scale)

    elif method == 'faithful':
        prior_precs = np.logspace(-2, 4, 7)
        nlls = []
        for prior_prec in prior_precs:
            set_seed(seed)
            model.reset_parameters()
            model, valid_perfs, valid_nlls = faithful_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs_val, beta=beta,
                 prior_structure='scalar', scheduler='cos', optimizer=optimizer, prior_prec_init=prior_prec, use_wandb=False
            )
            nlls.append(valid_nlls[-1])
            if use_wandb:
                wandb.run.summary[f'valnll/prior_prec_{prior_prec}'] = valid_nlls[-1]

        # choose best prior precision and rerun on combined train + validation set
        opt_prior_prec = prior_precs[np.nanargmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.nanmin(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')
        set_seed(seed)
        model.reset_parameters()
        model, _, _ = faithful_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=beta,
            scheduler='cos', optimizer=optimizer, prior_prec_init=opt_prior_prec, use_wandb=use_wandb
        )

        # evaluate
        wandb.run.define_metric('predictive', summary='mean')
        scale = ds_train.std_target
        test_mse = 0
        test_loglik = 0
        fs = list()
        reps = list()
        f_mus, f_stds = list(), list()
        N = len(test_loader.dataset)
        for i, (x, y) in enumerate(test_loader):
            with Timer('predictive', wandb=True, logger=True, step=i):
                with torch.no_grad():
                    f = model(x)
            mu = f[: , 0]
            std = torch.sqrt(f[: , 1])
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N
            fs.append(std.detach().cpu().numpy())
            reps.append(model.representation(x).cpu().numpy())
            f_mus.append(mu.detach().cpu() * scale)
            f_stds.append(std.detach().cpu() * scale)

    elif method == 'mcdropout':
        output_size = 2
        prior_precs = np.logspace(-2, 4, 7)
        nlls = []
        for prior_prec in prior_precs:
            if model_name_for_vi == 'mlp':
                model = MLP(
                    input_size=channels * pixels ** 2, width=500, depth=3, output_size=output_size,
                    activation=activation, head=head, head_activation=head_activation, skip_head=head is None, dropout=0.05
                ).to(device)

            elif model_name_for_vi == 'cnn':
                model = LeNetMCDrpt(
                    in_channels=channels, n_out=output_size, activation=activation, n_pixels=pixels,
                    head=head, head_activation=head_activation, skip_head=head is None, dropoutp=0.05
                ).to(device)
            
            else:
                raise ValueError('Invalid model for MCDO.')

            if double:
                model = model.double()

            model, valid_perfs, valid_nlls = mcdropout_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=0.0,
                prior_structure='scalar', scheduler='cos', optimizer=optimizer, use_wandb=False,
                )  # Beta 0.0 to have NLL
            nlls.append(valid_nlls[-1])

        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')

        if model_name_for_vi == 'mlp':
            model = MLP(
                input_size=channels * pixels ** 2, width=500, depth=3, output_size=output_size,
                activation=activation, head=head, head_activation=head_activation, skip_head=head is None, dropout=0.05
            ).to(device)

        elif model_name_for_vi == 'cnn':
            model = LeNetMCDrpt(
                in_channels=channels, n_out=output_size, activation=activation, n_pixels=pixels,
                head=head, head_activation=head_activation, skip_head=head is None, dropoutp=0.05
            ).to(device)

        if double:
            model = model.double()

        model, _, _ = mcdropout_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=0.0,
            scheduler='cos', optimizer=optimizer, use_wandb=use_wandb)

        # Evaluate the trained model on test set.
        wandb.define_metric('predictive', summary='mean')
        scale = ds_train.std_target
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        reps = list()
        f_mus, f_stds = list(), list()
        fs = list()
        model.eval()
        enable_dropout(model)
        for i, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                reps.append(model.representation(x).cpu().numpy())
            with Timer('predictive', wandb=True, logger=True, step=i):
                f_msamples = torch.stack([model(x) for k in range(5)], dim=1)
                mu = f_msamples[:, :, 0].mean(1)
                std = torch.sqrt(f_msamples[:, :, 1].mean(1))
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N
            f_mus.append(mu.detach().cpu() * scale)
            f_stds.append(std.detach().cpu() * scale)
            fs.append(std.detach().cpu().numpy())

    elif method == 'vi':
        output_size = 2
        prior_precs = np.logspace(-2, 4, 7)
        nlls = []
        confs = []
        for prior_prec in prior_precs:
            if model_name_for_vi == 'mlp':
                model = MLP(
                    input_size=channels * pixels ** 2, width=500, depth=3, output_size=output_size,
                    activation=activation, head=head, head_activation=head_activation, skip_head=head is None
                ).to(device)

            elif model_name_for_vi == 'cnn':
                model = LeNet(
                    in_channels=channels, n_out=output_size, activation=activation, n_pixels=pixels,
                    head=head, head_activation=head_activation, skip_head=head is None
                ).to(device)

            elif model_name_for_vi == 'resnet':
                model = ResNet(depth=20, in_planes=32, in_channels=3, n_out=output_size,
                               head=head, head_activation=head_activation, skip_head=head is None).to(device)

            # Make selected model Bayesian (using Bayesian Layers instead of deterministic ones)
            make_bayesian(model, prior_mu=vi_prior_mu, prior_sigma=1. / prior_prec,
                          posterior_mu_init=vi_posterior_mu_init, posterior_rho_init=vi_posterior_rho_init,
                          typeofrep=typeofrep)

            model.to(device)
            if double:
                model = model.double()

            model, valid_perfs, valid_nlls = vi_optimization(
                model, train_loader, valid_loader=valid_loader, lr=lr, lr_min=lr_min, n_epochs=n_epochs, beta=0.0,
                prior_structure='scalar', scheduler='cos', optimizer=optimizer, use_wandb=False,
                double=double)  # Beta 0.0 to have NLL
            nlls.append(valid_nlls[-1])

        opt_prior_prec = prior_precs[np.argmin(nlls)]
        if use_wandb:
            wandb.run.summary['prior_prec_opt'] = opt_prior_prec
            wandb.run.summary['valid/nll'] = np.min(nlls)
        logging.info(f'Best prior precision found: {opt_prior_prec}')

        if model_name_for_vi == 'mlp':
            model = MLP(
                input_size=channels * pixels ** 2, width=500, depth=3, output_size=output_size,
                activation=activation, head=head, head_activation=head_activation, skip_head=head is None
            ).to(device)

        elif model_name_for_vi == 'cnn':
            model = LeNet(
                in_channels=channels, n_out=output_size, activation=activation, n_pixels=pixels,
                head=head, head_activation=head_activation, skip_head=head is None
            ).to(device)

        elif model_name_for_vi == 'resnet':
            # if method in ['map', 'marglik', 'betanll']:
            model = ResNet(depth=20, in_planes=32, in_channels=3, n_out=output_size,
                            head=head, head_activation=head_activation, skip_head=head is None).to(device)

        # Make selected model Bayesian (using Bayesian Layers instead of deterministic ones)
        make_bayesian(model, prior_mu=vi_prior_mu, prior_sigma=1. / opt_prior_prec,
                      posterior_mu_init=vi_posterior_mu_init, posterior_rho_init=vi_posterior_rho_init,
                      typeofrep=typeofrep)
        model.to(device)
        if double:
            model = model.double()
        # model.reset_parameters()
        model, _, _ = vi_optimization(
            model, train_loader_full, lr=lr, lr_min=lr_min, n_epochs=n_epochs, prior_structure='scalar', beta=0.0,
            scheduler='cos', optimizer=optimizer, use_wandb=use_wandb, double=double)

        # Evaluate the trained model on test set.
        wandb.define_metric('predictive', summary='mean')
        scale = ds_train.std_target
        test_mse = 0
        test_loglik = 0
        N = len(test_loader.dataset)
        reps = list()
        f_mus, f_stds = list(), list()
        fs = list()
        for i, (x, y) in enumerate(test_loader):
            with torch.no_grad():
                reps.append(model.representation(x).cpu().numpy())
            # f = model(x)
            # mu = f[:, 0]
            # std = f[:,1]
            with Timer('predictive', wandb=True, logger=True, step=i):
                f_msamples = torch.stack([model(x) for k in range(5)], dim=1)
                mu = f_msamples[:, :, 0].mean(1)
                std = torch.sqrt(f_msamples[:, :, 1].mean(1))
            test_loglik += Normal(scale * mu, scale * std).log_prob(y.squeeze() * scale).sum().item() / N
            test_mse += (y.squeeze() - mu).square().sum() / N
            f_mus.append(mu.detach().cpu() * scale)
            f_stds.append(std.detach().cpu() * scale)
            fs.append(std.detach().cpu().numpy())

    else:
        raise ValueError('Invalid method')

    # Train logistic regression on top of learned features.
    treps = list()
    train_loader_full = TensorDataLoader(x_train, y_train, batch_size=batch_size, shuffle=False)
    for x, y in train_loader_full:
        treps.append(model.representation(x).cpu().numpy())
    treps = np.concatenate(treps, axis=0)
    scaler = StandardScaler()
    treps = scaler.fit_transform(treps)
    labels = ds_train.labels.numpy().flatten()
    clf = LogisticRegression(random_state=seed).fit(treps, labels)
    train_acc = clf.score(treps, labels)
    labels = ds_test.labels.cpu().numpy().flatten()
    reps = scaler.transform(np.concatenate(reps, axis=0))
    acc = clf.score(reps, labels)

    # true dist
    gt_dist = ds_test.ground_truth_distribution()
    # pred dist
    fmu, fstd = torch.cat(f_mus), torch.cat(f_stds)
    pr_dist = Normal(fmu, fstd)
    kl_div = kl_divergence(gt_dist, pr_dist).mean().item()
    test_rmse_rot = (fmu - ds_test.rotations).square().mean().sqrt().item()

    if likelihood == 'heteroscedastic':
        fs = np.concatenate(fs)
        aleatoric_corr = np.corrcoef(labels, fs)[0, 1]
    if method in ['marglik', 'map']:
        fse = np.concatenate(fse)
        predictive_corr = np.corrcoef(labels, fse)[0, 1]
    if use_wandb:
        wandb.run.summary['train/acc'] = train_acc
        wandb.run.summary['test/acc'] = acc
        wandb.run.summary['test/mse'] = ds_train.std_target ** 2 * test_mse
        wandb.run.summary['test/rmse'] = ds_train.std_target * sqrt(test_mse)
        wandb.run.summary['test/rmse_rot'] = test_rmse_rot
        wandb.run.summary['test/kl_div'] = kl_div
        wandb.run.summary['test/loglik'] = test_loglik
        if likelihood == 'heteroscedastic':
            wandb.run.summary['test/aleatoric_corr'] = aleatoric_corr
        if method in ['map', 'marglik']:
            wandb.run.summary['test/loglik_bayes'] = test_loglik_bayes
            wandb.run.summary['test/predictive_corr'] = predictive_corr
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
    parser.add_argument('--dataset', default='mnist', choices=IMAGE_DATASETS)
    parser.add_argument('--het_noise', default='label', choices=['label', 'rotation', 'neither'])
    parser.add_argument('--double', default=True, action=argparse.BooleanOptionalAction)
    # architecture
    parser.add_argument('--model', default='mlp', choices=['mlp', 'cnn', 'resnet', 'mlpmixer', 'vit'])
    parser.add_argument('--head', default='natural', choices=HEADS)
    parser.add_argument('--head_activation', default='softplus', choices=['exp', 'softplus'])
    parser.add_argument('--activation', default='relu', choices=ACTIVATIONS)
    # optimization (general)
    parser.add_argument('--lr', default=1e-1, type=float)
    parser.add_argument('--lr_min', default=1e-6, type=float, help='Cosine decay target')
    parser.add_argument('--n_epochs', default=300, type=int)
    parser.add_argument('--n_epochs_val', default=None, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--optimizer', default='Adam', help='Optimizer', choices=['Adam', 'SGD'])
    parser.add_argument('--method', default='map', help='Method', choices=['map', 'marglik', 'betanll', 'faithful', 'vi', 'mcdropout'])
    parser.add_argument('--beta', default=0.5, type=float)
    parser.add_argument('--likelihood', default='heteroscedastic', choices=['heteroscedastic', 'homoscedastic'])
    parser.add_argument('--prior_prec_init', default=1.0, type=float, help='Prior precision init or final for MAP.')
    # marglik-specific
    parser.add_argument('--approx', default='kron', choices=['full', 'kron', 'diag'])
    parser.add_argument('--lr_hyp', default=0.1, type=float)
    parser.add_argument('--lr_hyp_min', default=0.1, type=float)
    parser.add_argument('--n_epochs_burnin', default=10, type=int)
    parser.add_argument('--marglik_frequency', default=10, type=int)
    parser.add_argument('--n_hypersteps', default=50, help='Number of steps on every marglik estimate (partial grad accumulation)', type=int)
    parser.add_argument('--marglik_early_stopping', default=True, action=argparse.BooleanOptionalAction)
    # others
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--data_root', default='data/')
    parser.add_argument('--download_data', default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument('--use_wandb', default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument('--config', nargs='+')
    parser.add_argument('--vi-prior-mu', default=0.0, type=float)
    parser.add_argument('--vi-posterior-mu-init', default=0.0, type=float)
    parser.add_argument('--vi-posterior-rho-init', default=-3.0, type=float)
    parser.add_argument('--typeofrep', default="Flipout", choices=['Flipout', 'Reparameterization'])

    set_defaults_with_yaml_config(parser, sys.argv)
    args = vars(parser.parse_args())
    args.pop('config')
    if args['n_epochs_val'] is None:
        args['n_epochs_val'] = args['n_epochs']
    if args['method'] == 'map':
        # do not do marglik optimization
        args['n_epochs_burnin'] = args['n_epochs'] + 1
    if args['method'] != 'betanll':
        args['beta'] = 0.0
    if args['method'] == 'betanll':
        args['head'] = 'gaussian'
    if args['method'] == 'faithful':
        args['head'] = 'gaussian'
    if args['method'] == 'vi':
        args['head'] = 'gaussian'
    if args['method'] == 'mcdropout':
        args['head'] = 'gaussian'
    print(args)
    if args['use_wandb']:
        import uuid
        import copy
        tags = [args['dataset'], args['method']]
        config = copy.deepcopy(args)
        run_name = '-'.join(tags)
        run_name += '-' + str(uuid.uuid5(uuid.NAMESPACE_DNS, str(args)))[:4]
        config['method_name'] = '-'.join([args['method'], args['likelihood'][:3], args['head']])
        if 'betanll' in args['method']:
            config['method_name'] += f'-beta={args["beta"]}'
        load_dotenv()
        wandb.init(project='image-regression-final', entity='hetreg',
                   mode='online', config=config, name=run_name, tags=tags)
    main(**args)
