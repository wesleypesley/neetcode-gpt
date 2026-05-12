import torch
import torch.nn as nn
import math
from typing import List

class Solution:
    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        z = math.sqrt(2 / (fan_in + fan_out)) * torch.randn(fan_out, fan_in)
        return torch.round(z, decimals=4).numpy().tolist()

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        torch.manual_seed(0)
        z = math.sqrt(2 / fan_in) * torch.randn(fan_out, fan_in)
        return torch.round(z, decimals=4).numpy().tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        torch.manual_seed(0)
        stds = []

        # Generate ALL weight matrices first
        layers = []
        for i in range(num_layers):
            fan_in = input_dim if i == 0 else hidden_dim
            if init_type.lower() == 'kaiming':
                std = math.sqrt(2.0 / fan_in)
            elif init_type.lower() == 'xavier':
                std = math.sqrt(2.0 / (fan_in + hidden_dim))
            else:
                std = 1.0
            W = torch.randn(hidden_dim, fan_in) * std
            layers.append(W)

        # THEN generate input x
        inputs = torch.randn(input_dim)

        # Forward pass
        for W in layers:
            inputs = torch.relu(W @ inputs)
            stds.append(float(f"{inputs.std().item():.2f}"))

        return stds