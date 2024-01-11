import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from functools import partial
from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

from bayesian_torch.layers import LinearReparameterization
from asdfghjkl.operations import Bias, Scale

from blitz.modules import BayesianLinear, BayesianConv2d
from blitz.utils import variational_estimator

from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn


ACTIVATIONS = ['relu', 'selu', 'tanh', 'silu', 'gelu']
HEADS = ['natural', 'meanvar', 'gaussian']

def stable_softplus(input):
   return F.softplus(input) + 1e-8

def get_activation(act_str):
    if act_str == 'relu':
        return nn.ReLU
    elif act_str == 'tanh':
        return nn.Tanh
    elif act_str == 'selu':
        return nn.SELU
    elif act_str == 'silu':
        return nn.SiLU
    elif act_str == 'gelu':
        return nn.GELU
    else:
        raise ValueError('invalid activation')


def get_head_activation(act_str):
    if act_str == 'exp':
        return torch.exp
    elif act_str == 'softplus':
        return stable_softplus
    else:
        raise ValueError('invalid activation')


class NaturalHead(nn.Module):

    def __init__(self, activation='softplus') -> None:
        super().__init__()
        self.act_fn = get_head_activation(activation)

    def forward(self, input):
        return torch.stack([input[:, 0], -0.5 * self.act_fn(input[:, 1])], 1)


class GaussianHead(nn.Module):

    def __init__(self, activation='softplus') -> None:
        super().__init__()
        self.act_fn = get_head_activation(activation)

    def forward(self, input):
        f1, f2 = input[:, 0], self.act_fn(input[:, 1])
        return torch.stack([f1, f2], 1)

        
def get_head(head_str):
    if head_str == 'natural':
        return NaturalHead
    elif head_str == 'gaussian':
        return GaussianHead
    else:
        return nn.Identity


class NaturalReparamHead(nn.Module):
    # Transform mean-var into natural parameters
    def forward(self, input):
        f1, f2 = input[:, 0], input[:, 1]
        eta_1 = f1 / f2
        eta_2 = - 1 / (2 * f2)
        return torch.stack([eta_1, eta_2], 1)


class MLP(nn.Sequential):

    def __init__(self, input_size, width, depth, output_size, activation='gelu',
                 head='natural', head_activation='exp', skip_head=False, dropout=0.0):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)
        self.rep_layer = f'layer{depth}'

        self.add_module('flatten', nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=True))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', nn.Linear(in_size, out_size, bias=True))
                if dropout > 0.0:
                    self.add_module(f'dropout{i+1}', nn.Dropout(p=dropout))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', nn.Linear(hidden_sizes[-1], output_size, bias=True))
        if not skip_head:
            self.add_module('head', get_head(head)(activation=head_activation))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()

#    def init_parameters_vi(self):
#        for module in self.modules():
#            if isinstance(module, LinearReparameterization):
#                module.init_parameters()

def make_bayesian(model, prior_mu, prior_sigma, posterior_mu_init, posterior_rho_init, typeofrep):
    const_bnn_prior_parameters = {
        "prior_mu": prior_mu,
        "prior_sigma": prior_sigma,
        "posterior_mu_init": posterior_mu_init,
        "posterior_rho_init": posterior_rho_init,
        "type": typeofrep,  # Flipout or Reparameterization
        "moped_enable": False,  # True to initialize mu/sigma from the pretrained dnn weights
        "moped_delta": 0.5,
    }
    dnn_to_bnn(model, const_bnn_prior_parameters)
    return model

@variational_estimator
class MLPVI(nn.Sequential):

    def __init__(self, input_size, width, depth, output_size, activation='gelu',
                 head='natural', head_activation='exp', skip_head=False, priorsigma1=0.1, priorsigma2=0.4):
        super(MLPVI, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)
        self.priorsigma1 = priorsigma1
        self.priorsigma2 = priorsigma2
        self.rep_layer = f'layer{depth}'
        
        self.add_module('flatten', nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', BayesianLinear(self.input_size, output_size, bias=True, prior_sigma_1=self.priorsigma1, prior_sigma_2=self.priorsigma2))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i+1}', BayesianLinear(in_size, out_size, bias=True, prior_sigma_1=self.priorsigma1, prior_sigma_2=self.priorsigma2))
                self.add_module(f'{activation}{i+1}', act())
            self.add_module('out_layer', BayesianLinear(hidden_sizes[-1], output_size, bias=True, prior_sigma_1=self.priorsigma1, prior_sigma_2=self.priorsigma2))
        if not skip_head:
            self.add_module('head', get_head(head)(activation=head_activation))

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def init_parameters_vi(self):
        for module in self.modules():
            if isinstance(module, LinearReparameterization):
                module.init_parameters()
            if isinstance(module, BayesianLinear):
                module.reset_parameters()


class MLPFaithfulseq(nn.Sequential):
    def __init__(self, input_size, width, depth, output_size, activation='gelu',
                 head='natural', head_activation='exp'):
        super(MLPFaithfulseq, self).__init__()
        self.input_size = input_size
        self.width = width
        self.depth = depth
        hidden_sizes = depth * [width]
        self.activation = activation
        act = get_activation(activation)

        self.add_module('flatten', nn.Flatten())
        if len(hidden_sizes) == 0:  # i.e. when depth == 0.
            # Linear Model
            self.add_module('lin_layer', nn.Linear(self.input_size, output_size, bias=True))
        else:
            # MLP
            in_outs = zip([self.input_size] + hidden_sizes[:-1], hidden_sizes)
            for i, (in_size, out_size) in enumerate(in_outs):
                self.add_module(f'layer{i + 1}', nn.Linear(in_size, out_size, bias=True))
                self.add_module(f'{activation}{i + 1}', act())


class MLPFaithful(nn.Module):
    def __init__(self, input_size, width, depth, activation='gelu',
                 head='natural', head_activation='exp'):
        super(MLPFaithful, self).__init__()

        self.z_layer = MLPFaithfulseq(input_size, width, depth-1, output_size=width, activation=activation,
                 head=head, head_activation=head_activation)

        self.out_layer_mu = nn.Linear(width, 1, bias=True)
        self.out_layer_var = nn.Linear(width, 1, bias=True)

        self.head_layer = get_head(head)(activation=head_activation)

    def forward(self, input):
        z = self.z_layer(input)
        mu_z = self.out_layer_mu(z)
        var_z = self.out_layer_var(z.detach())     # Stop gradient here
        return self.head_layer(torch.cat([mu_z, var_z], dim=-1))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def representation(self, input):
        return self.z_layer(input).detach()


class MLPFaithfulBottleneck(nn.Module):
    def __init__(self, input_size, width, depth, bottleneck, activation='gelu',
                 head='natural', head_activation='exp'):
        super(MLPFaithfulBottleneck, self).__init__()

        self.z_layer = MLP(input_size, width, depth, output_size=bottleneck, activation=activation,
                 head=head, head_activation=head_activation, skip_head=True)
        self.out_layer_mu = nn.Linear(bottleneck, 1, bias=True)
        self.out_layer_var = nn.Linear(bottleneck, 1, bias=True)

        self.head_layer = get_head(head)(activation=head_activation)

    def forward(self, input):
        z = self.z_layer(input)
        mu_z = self.out_layer_mu(z)
        var_z = self.out_layer_var(z.detach())     # Stop gradient here
        return self.head_layer(torch.cat([mu_z, var_z], dim=-1))


    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class LeNet(nn.Sequential):

    def __init__(self, in_channels=1, n_out=10, activation='gelu', n_pixels=28,
                 head='natural', head_activation='exp', skip_head=False, only_rep=False):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = nn.Conv2d
        pool = nn.MaxPool2d
        self.rep_layer = 'lin2'
        flatten = nn.Flatten(start_dim=1)
        self.add_module('conv1', conv(in_channels, 6, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(6, 32, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(32, 256, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('lin1', torch.nn.Linear(256, 128))
        self.add_module('act4', act())
        self.add_module('lin2', torch.nn.Linear(128, 64))
        if not only_rep:
            self.add_module('act5', act())
            self.add_module('linout', torch.nn.Linear(64, n_out))
            if not skip_head:
                self.add_module('head', get_head(head)(activation=head_activation))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.reset_parameters()

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()

class LeNetMCDrpt(nn.Sequential):

    def __init__(self, in_channels=1, n_out=10, activation='gelu', n_pixels=28,
                 head='natural', head_activation='exp', skip_head=False, only_rep=False, dropoutp=0.05):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = nn.Conv2d
        pool = nn.MaxPool2d
        dropout = nn.Dropout(dropoutp)
        self.rep_layer = 'lin2'
        flatten = nn.Flatten(start_dim=1)
        self.add_module('conv1', conv(in_channels, 6, 5, 1))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('dropout1', dropout)
        self.add_module('conv2', conv(6, 32, mid_kernel_size, 1))
        self.add_module('act2', act())
        self.add_module('dropout2', dropout)
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(32, 256, 5, 1))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('dropout3', dropout)
        self.add_module('lin1', torch.nn.Linear(256, 128))
        self.add_module('act4', act())
        self.add_module('dropout4', dropout)
        self.add_module('lin2', torch.nn.Linear(128, 64))
        if not only_rep:
            self.add_module('act5', act())
            self.add_module('linout', torch.nn.Linear(64, n_out))
            if not skip_head:
                self.add_module('head', get_head(head)(activation=head_activation))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.reset_parameters()

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()

@variational_estimator
class LeNet_VI(nn.Sequential):

    def __init__(self, in_channels=1, n_out=10, activation='gelu', n_pixels=28,
                 head='natural', head_activation='exp', skip_head=False, only_rep=False,  priorsigma1=0.1, priorsigma2=0.4):
        super().__init__()
        mid_kernel_size = 3 if n_pixels == 28 else 5
        act = get_activation(activation)
        conv = BayesianConv2d
        pool = nn.MaxPool2d
        self.rep_layer = 'lin2'
        flatten = nn.Flatten(start_dim=1)
        self.add_module('conv1', conv(in_channels, 6, (5, 5), 1, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2))
        self.add_module('act1', act())
        self.add_module('pool1', pool(2))
        self.add_module('conv2', conv(6, 32, (mid_kernel_size, mid_kernel_size), 1, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2))
        self.add_module('act2', act())
        self.add_module('pool2', pool(2))
        self.add_module('conv3', conv(32, 256, (5,5), 1, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2))
        self.add_module('flatten', flatten)
        self.add_module('act3', act())
        self.add_module('lin1', BayesianLinear(256, 128, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2))
        self.add_module('act4', act())
        self.add_module('lin2', BayesianLinear(128, 64, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2))
        if not only_rep:
            self.add_module('act5', act())
            self.add_module('linout', BayesianLinear(64, n_out, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2))
            if not skip_head:
                self.add_module('head', get_head(head)(activation=head_activation))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                module.reset_parameters()

    def representation(self, input):
        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook
        handle = getattr(self, self.rep_layer).register_forward_hook(get_activation(self.rep_layer))
        self.forward(input)
        rep = activation[self.rep_layer]
        handle.remove()
        return rep.detach()


class LeNetFaithful(nn.Module):
    def __init__(self, in_channels, n_out, activation='gelu', n_pixels=28,
                 head='gaussian', head_activation='softplus'):
        super(LeNetFaithful, self).__init__()
        self.z_layer = LeNet(in_channels, n_out, activation, n_pixels, only_rep=True)
        self.out_layer_mu = nn.Linear(64, 1, bias=True)
        self.out_layer_var = nn.Linear(64, 1, bias=True)
        self.head_layer = get_head(head)(activation=head_activation)

    def forward(self, input):
        z = self.z_layer(input)
        mu_z = self.out_layer_mu(z)
        var_z = self.out_layer_var(z.detach())     # Stop gradient here
        return self.head_layer(torch.cat([mu_z, var_z], dim=-1))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def representation(self, input):
        return self.z_layer(input).detach()



def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv3x3Bayes(in_planes, out_planes, stride=1, priorsigma1=0.1, priorsigma2=0.4):
    """3x3 convolution with padding"""
    return BayesianConv2d(in_planes, out_planes, kernel_size=(3,3), stride=stride,
                     padding=1, bias=False, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2)


class FixupBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(FixupBasicBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = Bias()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3(planes, planes)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    FixupResnet-depth where depth is a `3 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, n_out=2, in_planes=16, in_channels=3, head='natural',
                 head_activation='exp', skip_head=False, only_rep=False):
        super(ResNet, self).__init__()
        assert (depth - 2) % 6 == 0, 'Invalid ResNet depth, has to conform to 6 * n + 2'
        layer_size = (depth - 2) // 6
        layers = 3 * [layer_size]
        self.num_layers = 3 * layer_size
        self.inplanes = in_planes
        self.conv1 = conv3x3(in_channels, in_planes)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock, in_planes, layers[0])
        self.layer2 = self._make_layer(FixupBasicBlock, in_planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(FixupBasicBlock, in_planes * 4, layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.bias2 = Bias()
        if not only_rep:
            self.fc = nn.Linear(in_planes * 4, n_out)
        else:
            self.fc = None

        if not skip_head:
            self.head = get_head(head)(activation=head_activation)
        else:
            self.head = None

        self.reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.bias2(x)
        if self.fc is not None:
            x = self.fc(x)

        if self.head is not None:
            x = self.head(x)

        return x

    def representation(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.relu(self.bias1(x))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = self.flatten(x)
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, Scale, Bias)):
                module.reset_parameters()

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight,
                                mean=0,
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)

@variational_estimator
class FixupBasicBlock_VI(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, priorsigma1=0.1, priorsigma2=0.4):
        super(FixupBasicBlock_VI, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.bias1a = Bias()
        self.conv1 = conv3x3Bayes(inplanes, planes, stride, priorsigma1=priorsigma1, priorsigma2=priorsigma2)
        self.bias1b = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = Bias()
        self.conv2 = conv3x3Bayes(planes, planes, priorsigma1=priorsigma1, priorsigma2=priorsigma2)
        self.scale = Scale()
        self.bias2b = Bias()
        self.downsample = downsample

    def forward(self, x):
        identity = x

        biased_x = self.bias1a(x)
        out = self.conv1(biased_x)
        out = self.relu(self.bias1b(out))

        out = self.conv2(self.bias2a(out))
        out = self.bias2b(self.scale(out))

        if self.downsample is not None:
            identity = self.downsample(biased_x)
            identity = torch.cat((identity, torch.zeros_like(identity)), 1)

        out += identity
        out = self.relu(out)

        return out

@variational_estimator
class ResNet_VI(nn.Module):
    """
    FixupResnet-depth where depth is a `3 * 2 * n + 2` with `n` blocks per residual layer.
    The two added layers are the input convolution and fully connected output.
    """

    def __init__(self, depth, n_out=2, in_planes=16, in_channels=3, head='natural',
                 head_activation='exp', skip_head=False, only_rep=False, priorsigma1=0.1, priorsigma2=0.4):
        super(ResNet_VI, self).__init__()
        assert (depth - 2) % 6 == 0, 'Invalid ResNet depth, has to conform to 6 * n + 2'
        layer_size = (depth - 2) // 6
        layers = 3 * [layer_size]
        self.num_layers = 3 * layer_size
        self.inplanes = in_planes
        self.conv1 = conv3x3Bayes(in_channels, in_planes, priorsigma1=priorsigma1, priorsigma2=priorsigma2)
        self.bias1 = Bias()
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(FixupBasicBlock_VI, in_planes, layers[0], priorsigma1=priorsigma1, priorsigma2=priorsigma2)
        self.layer2 = self._make_layer(FixupBasicBlock_VI, in_planes * 2, layers[1], stride=2, priorsigma1=priorsigma1, priorsigma2=priorsigma2)
        self.layer3 = self._make_layer(FixupBasicBlock_VI, in_planes * 4, layers[2], stride=2, priorsigma1=priorsigma1, priorsigma2=priorsigma2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.bias2 = Bias()
        if not only_rep:
            self.fc = BayesianLinear(in_planes * 4, n_out, prior_sigma_1=priorsigma1, prior_sigma_2=priorsigma2)
        else:
            self.fc = None

        if not skip_head:
            self.head = get_head(head)(activation=head_activation)
        else:
            self.head = None

        self.reset_parameters()

    def _make_layer(self, block, planes, blocks, stride=1, priorsigma1=0.1, priorsigma2=0.4):
        downsample = None
        if stride != 1:
            downsample = nn.AvgPool2d(1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, priorsigma1=priorsigma1, priorsigma2=priorsigma2))
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(planes, planes, priorsigma1=priorsigma1, priorsigma2=priorsigma2))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(self.bias1(x))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.bias2(x)
        if self.fc is not None:
            x = self.fc(x)

        if self.head is not None:
            x = self.head(x)

        return x

    def representation(self, x):
        with torch.no_grad():
            x = self.conv1(x)
            x = self.relu(self.bias1(x))
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            x = self.flatten(x)
        return x

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear, Scale, Bias)):
                module.reset_parameters()

        for m in self.modules():
            if isinstance(m, FixupBasicBlock):
                nn.init.normal_(m.conv1.weight,
                                mean=0,
                                std=np.sqrt(2 / (m.conv1.weight.shape[0] * np.prod(m.conv1.weight.shape[2:]))) * self.num_layers ** (-0.5))
                nn.init.constant_(m.conv2.weight, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0)
                nn.init.constant_(m.bias, 0)


class ResNetFaithful(nn.Module):
    def __init__(self, depth, n_out=2, in_planes=16, in_channels=3,
                 head='gaussian', head_activation='softplus'):
        super(ResNetFaithful, self).__init__()
        self.z_layer = ResNet(depth, n_out, in_planes, in_channels, head=None, skip_head=True, only_rep=True)
        self.out_layer_mu = nn.Linear(in_planes * 4, 1, bias=True)
        self.out_layer_var = nn.Linear(in_planes * 4, 1, bias=True)
        self.head_layer = get_head(head)(activation=head_activation)

    def forward(self, input):
        z = self.z_layer(input)
        mu_z = self.out_layer_mu(z)
        var_z = self.out_layer_var(z.detach())     # Stop gradient here
        return self.head_layer(torch.cat([mu_z, var_z], dim=-1))

    def reset_parameters(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

        self.z_layer.reset_parameters()

    def representation(self, input):
        return self.z_layer.representation(input).detach()


if __name__ == '__main__':
    from hetreg.image_datasets import (
        RotationRegressionCIFAR10, RotationRegressionMNIST, transform_mnist, transform_cifar
    )
    from hetreg.utils import dataset_to_tensors, TensorDataLoader
    ds = RotationRegressionMNIST(
        root='data/', train=False, transform=transform_mnist, download=True
    )
    x, y = dataset_to_tensors(ds, device='cpu')

    mlp = MLP(28 * 28, 500, 2, 2)
    cnn = LeNet(in_channels=1, n_out=2)
    loader = TensorDataLoader(x, y, batch_size=512, shuffle=True)
    for x, y in loader:
        print(x.shape, y.shape, x.dtype, y.dtype, mlp(x).shape, cnn(x).shape)

    ds = RotationRegressionCIFAR10(
        root='data/', train=False, download=True, max_rotation=90, transform=transform_cifar
    )
    x, y = dataset_to_tensors(ds, device='cpu')

    mlp = MLP(32*32*3, 500, 2, 2)
    cnn = LeNet(in_channels=3, n_pixels=32, n_out=2)
    resnet = ResNet(depth=8)
    loader = TensorDataLoader(x, y, batch_size=512, shuffle=True)
    for x, y in loader:
        print(x.shape, y.shape, x.dtype, y.dtype, mlp(x).shape, cnn(x).shape, resnet(x).shape)
