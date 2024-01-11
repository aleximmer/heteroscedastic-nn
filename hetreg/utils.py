from math import ceil, sqrt, log
import numpy as np
import torch
from torch import nn
import wandb
from time import time
import logging

from laplace import FullLaplace, KronLaplace, DiagLaplace, FunctionalLaplace


class TensorDataLoader:
    """Combination of torch's DataLoader and TensorDataset for efficient batch
    sampling and adaptive augmentation on GPU.
    """

    def __init__(self, x, y, transform=None, transform_y=None, batch_size=500,
                 data_factor=1, shuffle=False, detach=True):
        assert x.size(0) == y.size(0), 'Size mismatch'
        self.x = x
        self.y = y
        self.device = x.device
        self.data_factor = data_factor
        self.n_data = y.size(0)
        if batch_size < 0:
            self.batch_size = self.x.size(0)
        else:
            self.batch_size = batch_size
        self.n_batches = ceil(self.n_data / self.batch_size)
        self.shuffle = shuffle
        identity = lambda x: x
        self.transform = transform if transform is not None else identity
        self.transform_y = transform_y if transform_y is not None else identity
        self._detach = detach

    def __iter__(self):
        if self.shuffle:
            permutation = torch.randperm(self.n_data, device=self.device)
            self.x = self.x[permutation]
            self.y = self.y[permutation]
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        start = self.i_batch * self.batch_size
        end = start + self.batch_size
        if self._detach:
            x = self.transform(self.x[start:end]).detach()
        else:
            x = self.transform(self.x[start:end])
        y = self.transform_y(self.y[start:end])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return self.n_batches

    def attach(self):
        self._detach = False
        return self

    def detach(self):
        self._detach = True
        return self

    @property
    def dataset(self):
        return DatasetDummy(self.n_data * self.data_factor)


class SubsetTensorDataLoader(TensorDataLoader):

    def __init__(self, x, y, transform=None, transform_y=None, subset_size=500,
                 data_factor=1, detach=True):
        self.subset_size = subset_size
        super().__init__(x, y, transform, transform_y, batch_size=subset_size,
                         data_factor=data_factor, shuffle=True, detach=detach)
        self.n_batches = 1  # -> len(loader) = 1

    def __iter__(self):
        self.i_batch = 0
        return self

    def __next__(self):
        if self.i_batch >= self.n_batches:
            raise StopIteration

        sod_indices = np.random.choice(self.n_data, self.subset_size, replace=False)
        if self._detach:
            x = self.transform(self.x[sod_indices]).detach()
        else:
            x = self.transform(self.x[sod_indices])
        y = self.transform_y(self.y[sod_indices])
        self.i_batch += 1
        return (x, y)

    def __len__(self):
        return 1

    @property
    def dataset(self):
        return DatasetDummy(self.subset_size * self.data_factor)


class DatasetDummy:
    def __init__(self, N):
        self.N = N

    def __len__(self):
        return int(self.N)


def dataset_to_tensors(dataset, indices=None, device='cuda'):
    if indices is None:
        indices = range(len(dataset))  # all
    xy_train = [dataset[i] for i in indices]
    x = torch.stack([e[0] for e in xy_train]).to(device)
    y = torch.stack([torch.tensor(e[1]) for e in xy_train]).unsqueeze(-1).to(device)
    return x, y


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        import torch.backends.cudnn as cudnn
        torch.cuda.manual_seed(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False


def get_laplace_approximation(structure):
    if structure == 'full':
        return FullLaplace
    elif structure == 'kron':
        return KronLaplace
    elif structure == 'diag':
        return DiagLaplace
    elif structure == 'kernel':
        return FunctionalLaplace


def wandb_log_prior(prior_prec, prior_structure, model):
    prior_prec = prior_prec.detach().cpu().numpy().tolist()
    if prior_structure == 'scalar':
        wandb.log({'hyperparams/prior_prec': prior_prec[0]}, commit=False)
    elif prior_structure == 'layerwise':
        log = {f'hyperparams/prior_prec_{n}': p for p, (n, _) in
               zip(prior_prec, model.named_parameters())}
        wandb.log(log, commit=False)
    elif prior_structure == 'diagonal':
        hist, edges = prior_prec.data.cpu().histogram(bins=64)
        log = {f'hyperparams/prior_prec': wandb.Histogram(
            np_histogram=(hist.numpy().tolist(), edges.numpy().tolist())
        )}
        wandb.log(log, commit=False)


def wandb_log_parameter_norm(model):
    for name, param in model.named_parameters():
        avg_norm = (param.data.flatten() ** 2).sum().item() / np.prod(param.data.shape)
        wandb.log({f'params/{name}': avg_norm}, commit=False)


class Timer:
    def __init__(self, name, wandb=True, logger=False, step=None) -> None:
        self.logger = logger
        self.name = name
        self.step = step
        self.wandb = wandb

    def __enter__(self):
        self.start_time = time()

    def __exit__(self, *args, **kwargs):
        elapsed_time = time() - self.start_time
        msg = f'{self.name} took {elapsed_time:.3f}s'
        if self.wandb:
            wandb.log({f'timing/{self.name}_sec': elapsed_time})
        if self.logger:
            logging.info(msg)
        else:
            print(msg)
