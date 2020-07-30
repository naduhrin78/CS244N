#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2019-20: Homework 5
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """
    CNN for combining character embeddings.
    """
    def __init__(self, char_embed_size, word_embed_size, kernel_size=5, padding=1):
        """ Init CNN Layer.

        @param embed_size (int): Embedding size for character (dimensionality)
        @param kernel_size (int): Size of kernel.
        @param padding (int): Padding for conv1d layer.
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=char_embed_size, out_channels=word_embed_size, kernel_size=kernel_size
                              , padding=padding)
        # nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.uniform_(self.conv.bias)

        self.relu = nn.ReLU()
        self.maxpool = nn.AdaptiveMaxPool1d(output_size=1)
        # self.maxpool = nn.MaxPool1d(kernel_size=21 - kernel_size + 1)
        
    def forward(self, x_rshp: torch.Tensor) -> torch.Tensor:
        """ Takes a mini-batch of x_rshp and returns a tensor corresponding to conv operation output.

        @param x_rshp (torch.Tensor): Tensor returned from embedding layer of shape (embed_size, max_word_length).

        @returns x_conv_out (torch.Tensor): A tensor of shape (embed_size, ) representing output of conv layer.
        """
        x_conv = self.conv(x_rshp)
        x_conv_out = self.maxpool(self.relu(x_conv)).squeeze(-1)
        return x_conv_out
