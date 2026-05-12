import numpy as np

class Solution:
    def forward(self, x: np.ndarray, weights: list, biases: list) -> np.ndarray:
        h = x
        for i in range(len(weights)):
            z = h @ weights[i] + biases[i]
            if i < len(weights) - 1:  # hidden layers → ReLU
                h = np.maximum(0, z)
            else:                      # output layer → no activation
                h = z
        return np.round(h, 5)