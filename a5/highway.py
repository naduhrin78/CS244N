#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class Highway(nn.Module):
    """
    A highway network.
    """
    def __init__(self, embed_size):
        """ Init Highway Layer.

        @param embed_size (int): Embedding size for character (dimensionality)
        """
        super(Highway, self).__init__()
        self.embed_size = embed_size

        self.W_proj = nn.Linear(in_features=embed_size, out_features=embed_size)
        # nn.init.xavier_uniform_(self.W_proj.weight)
        # nn.init.uniform_(self.W_proj.bias)

        self.W_gate = nn.Linear(in_features=embed_size, out_features=embed_size)
        # nn.init.xavier_uniform_(self.W_gate.weight)
        # nn.init.uniform_(self.W_gate.bias)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_conv_out: torch.Tensor) -> torch.Tensor:
        """ Takes a mini-batch of x_conv_out and returns a tensor corresponding to highway operation output.

        @param x_conv_out (torch.Tensor): Tensor returned from convolutional network of shape (embed_size, ).

        @returns x_highway (torch.Tensor): A tensor of shape (embed_size, ) representing output of highway network.
        """
        x_proj = self.relu(self.W_proj(x_conv_out))
        x_gate = self.sigmoid(self.W_gate(x_conv_out))
        x_highway = x_gate * x_proj + (1-x_gate) * x_conv_out
        return x_highway
