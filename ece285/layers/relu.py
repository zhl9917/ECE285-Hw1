from .base_layer import BaseLayer
import numpy as np


class ReLU(BaseLayer):
    def __init__(self):
        self.cache = None

    def forward(self, input_x: np.ndarray):
        # TODO: Implement RELU activation function forward pass
        a = input_x>=0
        output = input_x * a
        # Store the input in cache, required for backward pass
        self.cache = input_x.copy()
        
        return output

    def backward(self, dout):
        # Load the input from the cache
        x_temp = self.cache
        # Calculate gradient for RELU
        dx = np.array((x_temp>=0),dtype=int)*dout
        return dx