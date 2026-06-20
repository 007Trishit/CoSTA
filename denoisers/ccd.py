#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from denoisers.multi_conv import MultiConv2d
from denoisers.linear_spline import LinearSpline
from denoisers.spline_module import LinearSpline as SplineModule
import math


def safe_inv(x):
    mask = x == 0
    x_inv = x**(-1)
    x_inv[mask] = 0
    return x_inv


class CCD(nn.Module):
    def __init__(self, param_multi_conv, param_spline_activation, param_spline_scaling, **kwargs):
        """
        
        Args:
            in_nc (int): Input channels
            out_nc (int): Intermediate channels
            kernel_size (int): Kernel size for convolutional layers
            stride (int): Stride for convolutional layers
            activation (str): Activation function to use
            step_size_type (str): Type of step size calculation
            use_multi_conv (bool): Whether to use multiple convolutions
            **kwargs: Additional keyword arguments for model configuration
        """
        super().__init__()

        self.tau = nn.Parameter(torch.tensor(0.001))
        self.step_size_type = kwargs.get('step_size_type', 't0')

        self.num_channels = param_multi_conv["num_channels"][-1]

        param_spline_activation['num_activations'] = self.num_channels
        self.activation = LinearSpline(**param_spline_activation)

        self.scaling = kwargs.get('scaling', False)
        self.blind = kwargs.get('blind', False)

        if self.scaling:
            param_spline_scaling['num_activations'] = self.num_channels
            self.spline_scaling = SplineModule(**param_spline_scaling)

        self.conv_layer = MultiConv2d(**param_multi_conv)

        self.bias = None
        if kwargs.get('bias', False):
            fan_in = self.conv_layer.conv_layers[-1].in_channels * \
                self.conv_layer.conv_layers[-1].kernel_size[-1] ** 2
            bound = 1 / math.sqrt(fan_in)
            self.bias = nn.Parameter(
                torch.empty(self.num_channels).uniform_(-bound, bound)
            )

        # Initialize Q parameters: one per output channel for each convolution layer
        self.Q = nn.ParameterList([
            nn.Parameter(torch.randn(conv.out_channels))
            for conv in self.conv_layer.conv_layers
        ])

    def compute_t(self):

        T = []
        for i, conv in enumerate(self.conv_layer.conv_layers):
            weight = conv.weight
            ktk = F.conv2d(weight, weight, padding=weight.shape[-1] - 1)
            ktk = torch.abs(ktk)
            q = torch.exp(self.Q[i]).reshape(-1, 1, 1, 1)
            q_inv = torch.exp(-self.Q[i]).reshape(-1, 1, 1, 1)
            t = (q_inv * ktk * q).sum((1, 2, 3)).sqrt()
            T.append(safe_inv(t).reshape(1, -1, 1, 1))
        return T

    def grad(self, x, sigma=None):
        """ Gradient of the loss at location x."""

        if not isinstance(sigma, torch.Tensor):
            sigma = sigma * torch.ones(1, 1, 1, 1)
        sigma = sigma.to(x.device)

        if not self.blind:
            noise_map = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3),
                                           device=x.device, dtype=x.dtype)
            y = torch.cat((x, noise_map), dim=1)
        else:
            y = x.clone()

        T = self.compute_t()

        for i, conv in enumerate(self.conv_layer.conv_layers):
            weight = conv.weight
            bias = self.bias if i == len(
                self.conv_layer.conv_layers) - 1 else None
            y = F.conv2d(y, weight, bias=bias, dilation=conv.dilation,
                         padding=conv.padding, groups=conv.groups, stride=conv.stride)

            y = T[i] * y

        if self.scaling:
            scaling = self.get_scaling(sigma)
            y = y * scaling

        # activation
        y = self.activation(y)

        if self.scaling:
            y = y / scaling

        for i, conv in enumerate(reversed(self.conv_layer.conv_layers)):
            y = T[-i-1] * y
            weight = conv.weight
            y = F.conv_transpose2d(y, weight, bias=None, padding=conv.padding,
                                   groups=conv.groups, dilation=conv.dilation, stride=conv.stride)

        if not self.blind:
            y = y[:, :-1, :, :]
        y = y + self.tau * x
        return (y)

    def forward(self, x, sigma=None):
        """
        Forward pass of CCD.
        
        Args:
            x (torch.Tensor): Input image [B, C, H, W]
            sigma (torch.Tensor, optional): Noise level map [B, 1, 1, 1]
                                         Required if not blind
        Returns:
            torch.Tensor: Denoised image
        """

        grad_x = self.grad(x, sigma=sigma)

        gamma = 1 if self.step_size_type == 't0' \
            else 2 / (1 + 2 * self.tau) if self.step_size_type == 't1' \
            else 1 / (1 + 2 * self.tau) if self.step_size_type == 't2' \
            else 1 / (1 + self.tau)

        x = x - gamma * grad_x
        # Remove noise level map
        return x

    def get_scaling(self, sigma=None):
        """Compute scaling factor based on sigma and channel"""
        eps = 1e-5
        return (torch.exp(self.spline_scaling(torch.tile(sigma, (1, self.num_channels, 1, 1)))) / (sigma + eps))

    def integrate_activation(self, x, sigma=None):

        if self.scaling:
            scaling = self.get_scaling(sigma)
        else:
            scaling = 1

        x = x * scaling

        y = self.activation.integrate(x)

        y = y / scaling / scaling

        return y

    def potential(self, x, sigma):
        s = x.shape
        # first multi convolution layer

        if not isinstance(sigma, torch.Tensor):
            sigma = sigma * torch.ones(1, 1, 1, 1)
        sigma = sigma.to(x.device)

        if not self.blind:
            noise_map = sigma * torch.ones(x.size(0), 1, x.size(2), x.size(3),
                                           device=x.device, dtype=x.dtype)
            y = torch.cat((x, noise_map), dim=1)

        else:
            y = x.clone()

        y = self.conv_layer(y)
        # activation
        y = self.integrate_activation(y, sigma)

        # Sum over C, H, W dimensions -> shape [B]
        potential_main = torch.sum(y, dim=tuple(range(1, len(s))))

        # Per-sample L2 norm squared -> shape [B]
        reg_term = 0.5 * self.tau * \
            torch.norm(x.view(x.shape[0], -1), p=2, dim=-1) ** 2

        return potential_main + reg_term
