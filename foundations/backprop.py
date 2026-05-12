import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        # Forward pass first
        z = np.dot(x, w) + b
        y_hat = 1 / (1 + np.exp(-z))

        # Chain rule pieces
        dL_dyhat = y_hat - y_true                  # error
        dyhat_dz = y_hat * (1 - y_hat)             # sigmoid derivative
        
        # Gradients
        dL_dw = dL_dyhat * dyhat_dz * x            # array, one per weight
        dL_db = dL_dyhat * dyhat_dz                # scalar

        return (np.round(dL_dw, 5), round(float(dL_db), 5))