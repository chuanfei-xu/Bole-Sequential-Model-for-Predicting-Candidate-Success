# -*- coding: utf-8 -*-
"""
Basic components for the model architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    """Feed-Forward Network with GELU activation"""
    def __init__(self, d_in_model, d_diff=768*4, d_out_model=768, drop_prob=0.1):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(d_in_model, d_diff)
        self.fc2 = nn.Linear(d_diff, d_out_model)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

