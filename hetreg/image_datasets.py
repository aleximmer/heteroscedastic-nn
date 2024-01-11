import torch
from torch.distributions import Normal
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms.functional import rotate
import torchvision.transforms as transforms

IMAGE_DATASETS = ['mnist', 'cifar10', 'fmnist']


transform_mnist = transforms.ToTensor()
mean = [x / 255 for x in [125.3, 123.0, 113.9]]
std = [x / 255 for x in [63.0, 62.1, 66.7]]
transform_cifar = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean, std)]
)


def get_dataset(name, root, seed, het_noise='label', download=False):
    """Get rotated image regression dataset variants.

    Parameters
    ----------
    name : str {'mnist', 'fmnist', 'cifar10'}
    root : str or pathlib.Path
    seed : int
    het_noise_label : str {'label', 'rotation', 'neither'}
        heteroscedastic noise dependent linearly on label, rotation, or neither
    download : bool, optional

    Returns
    -------
    tuple of train, test each torch.utils.data.Dataset
    """
    noise_scale = 1.0 if het_noise == 'rotation' else 10.0
    kwargs = {
        'root': root, 'download': download, 'seed': seed, 'het_noise': het_noise,
        'noise_scale': noise_scale
    }
    if name == 'mnist':
        kwargs['transform'] = transform_mnist
        ds_train = RotationRegressionMNIST(train=True, **kwargs)
        ds_test = RotationRegressionMNIST(
            train=False, std_target=ds_train.std_target, **kwargs)
    elif name == 'fmnist':
        kwargs['transform'] = transform_mnist
        ds_train = RotationRegressionFMNIST(train=True, **kwargs)
        ds_test = RotationRegressionFMNIST(
            train=False, std_target=ds_train.std_target, **kwargs)
    elif name == 'cifar10':
        kwargs['transform'] = transform_cifar
        ds_train = RotationRegressionCIFAR10(train=True, **kwargs)
        ds_test = RotationRegressionCIFAR10(
            train=False, std_target=ds_train.std_target, **kwargs)
    return ds_train, ds_test


class ImageRotationDatasetMixin:
    def __init__(self, root='data/', train=True, transform=None, target_transform=None,
                 download=False, max_rotation=90.0, seed=7, het_noise='label',
                 std_target=None, noise_scale=10.0):
        super(ImageRotationDatasetMixin, self).__init__(
            root=root, train=train, transform=transform,
            target_transform=target_transform, download=download
        )
        torch.manual_seed(seed + (1 if train else 0))
        rotations = (torch.rand(self.data.shape[0]) - 0.5) * 2 * max_rotation
        three_channel = self.data.shape[-1] == 3
        if three_channel:
            self.data = torch.from_numpy(self.data).transpose(1, 3).transpose(2, 3)
        # rotate each image by sampled rotation amount
        self.data = torch.cat([rotate(self.data[i].unsqueeze(0), rotations[i].item()) for i in range(self.data.shape[0])])
        if three_channel:
            self.data = self.data.transpose(2, 3).transpose(1, 3).numpy()
        # dependent on the label, sample Gaussian noise to the rotation target
        if not torch.is_tensor(self.targets):
            self.labels = torch.tensor(self.targets)  # original cifar10 labels in [0, 9]
        else:
            self.labels = self.targets
        if het_noise == 'neither':  # homoscedastic with noise_scale = std
            scale = torch.ones_like(self.labels) * noise_scale
        elif het_noise == 'label':  # heteroscedastic with noise_scale dependent on labels
            labels = self.labels.float()
            # noise scale is in range [noise_scale/10, 10*noise_scale] so typically [0.1, 10]
            scale = (labels + 0.1 * (labels + 1)) * noise_scale
        elif het_noise == 'rotation':  # heteroscedastic with noise_scale dependent on sampled rotation
            # noise scale is in range [0.1, sqrt(max_rotation)] * noise_scale so typically  [0.1, 10]
            scale = torch.clamp(torch.sqrt(rotations.abs()), min=0.1) * noise_scale
        else:
            raise ValueError('Invalid het_noise type')
        self.targets = rotations + torch.randn_like(rotations) * scale
        if std_target is None:
            self.std_target = torch.std(self.targets)
        else:
            self.std_target = std_target
        self.targets = self.targets / self.std_target
        self.rotations = rotations
        self.scale = scale

    def ground_truth_distribution(self):
        return Normal(self.rotations, self.scale)

    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        target = self.targets[index]  # override to avoid casts to int etc.
        return img, target


class RotationRegressionMNIST(ImageRotationDatasetMixin, MNIST):
    pass

class RotationRegressionFMNIST(ImageRotationDatasetMixin, FashionMNIST):
    pass

class RotationRegressionCIFAR10(ImageRotationDatasetMixin, CIFAR10):
    pass


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    ds = RotationRegressionMNIST(
        root='data/', train=False, download=True, transform=transform_mnist, seed=1,
        het_noise='neither', noise_scale=1.0)
    # plot some random training images
    f, axarr = plt.subplots(1, 10, figsize=(20, 6))
    for i in range(10):
        idx = torch.randint(len(ds), size=(1,)).item()
        axarr[i].imshow(ds[idx][0][0], cmap='gray_r')
        label = ds.labels[idx].item()
        rot = ds.rotations[idx].item()
        scale = ds.scale[idx].item()
        axarr[i].set_title(f'{label}; {rot:.2f} + {scale:.2f}; {ds.targets[idx].item()*ds.std_target:.2f}', fontsize=8)
    plt.show()

    # same for cifar10:
    ds = RotationRegressionCIFAR10(root='data/', train=False, download=True, max_rotation=100, transform=transform_cifar)
    f, axarr = plt.subplots(1, 10, figsize=(20, 6))
    for i in range(10):
        idx = torch.randint(len(ds), size=(1,)).item()
        label = ds.labels[idx].item()
        rot = ds.rotations[idx].item()
        scale = ds.scale[idx].item()
        axarr[i].imshow(ds[idx][0].transpose(0, 2).transpose(0, 1))
        axarr[i].set_title(f'{label}; {rot:.2f} + {scale:.2f}; {ds.targets[idx].item()*ds.std_target:.2f}', fontsize=8)
    plt.show()
