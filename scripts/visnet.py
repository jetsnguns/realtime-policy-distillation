from typing import Sequence, Optional

from ray.rllib.models.torch.misc import normc_initializer, valid_padding, \
    SlimConv2d, SlimFC

import torch as t
from torch import nn


class VisNet(nn.Module):
    def __init__(self, obs_space_shape, out_dims: int, filters=None, hiddens=512):
        super().__init__()

        if filters is None:
            filters = [
                [16, [8, 8], 4],
                [32, [4, 4], 2],
                [256, [11, 11], 1],
            ]

        activation = nn.ReLU

        layers = []
        (w, h, in_channels) = obs_space_shape
        in_size = [w, h]

        for out_channels, kernel, stride in filters[:-1]:
            padding, out_size = valid_padding(in_size, kernel,
                                              [stride, stride])
            layers.append(
                SlimConv2d(
                    in_channels,
                    out_channels,
                    kernel,
                    stride,
                    padding,
                    activation_fn=activation))
            in_channels = out_channels
            in_size = out_size

        out_channels, kernel, stride = filters[-1]
        layers.append(
            SlimConv2d(
                in_channels,
                out_channels,
                kernel,
                stride,
                None,
                activation_fn=activation))
        self._convs = nn.Sequential(*layers)

        self._hidden = SlimFC(
            out_channels, hiddens, initializer=nn.init.xavier_uniform_)

        self._logits = SlimFC(
            hiddens, out_dims, initializer=nn.init.xavier_uniform_)
        # Holds the current "base" output (before logits layer).
        self._features = None

    def forward(self, x):
        self._features = self._hidden_layers(x)
        logits = self._logits(self._features)
        return logits

    def _hidden_layers(self, obs):
        #res = self._convs(obs.permute(0, 3, 1, 2))  # switch to channel-major
        #res = res.squeeze(3)
        #res = res.squeeze(2)
        res = self._convs(obs)
        #print("res.shape = ", res.shape)
        res = res.squeeze(3)
        res = res.squeeze(2)
        #print("res.shape = ", res.shape)
        res = self._hidden(res)
        return res
