import random

from typing import Callable, List

try:
    from lib.activation import *
    from lib.helpers import *
except ModuleNotFoundError:
    from activation import *
    from helpers import *

class Layer:
    def __init__(self, nodes: int, prev_nodes: int) -> None:
        """
        Initializes a layer with a given number of nodes and a given number of weights.
        :param nodes: The number of nodes in the layer
        :param prev_nodes: The number of nodes in the previous layer.
        """
        LOWER_BOUND_U, UPPER_BOUND_U = -.2, .2
        self.prev_nodes = prev_nodes
        self.nodes = nodes
        self.weights = [[random.uniform(LOWER_BOUND_U, UPPER_BOUND_U) for _ in range(prev_nodes)] for _ in range(nodes)]
        self.bias = random.uniform(LOWER_BOUND_U, UPPER_BOUND_U)

    def __str__(self) -> str:
        """
        Returns a string representation of the layer.
        :return: A string representation of the layer.
        """
        return f"Layer(nodes={self.nodes}, bias={self.bias}, weights={self.weights})"

    def __repr__(self) -> str:
        """
        Returns a string representation of the layer.
        :return: A string representation of the layer.
        """
        return f"Layer(nodes={self.nodes}, bias={self.bias}, weights={self.weights})"
    
    def __len__(self) -> int:
        """
        Returns the number of nodes in the layer.
        :return: The number of nodes in the layer.
        """
        return self.nodes

    def forward_propagate(self, input_data: List[float], activation_function: Callable) -> List[float]:
        """
        Propagates the values in the previous layer through the layer. Handles laysrs with different number of nodes.
        :param input_data: The activations to carry through.
        :return: A list of the values in the layer.
        """ 
        assert len(input_data) == self.prev_nodes, "Error in forward propagation. The length of the data does not match the previous layer nodes."

        return [activation_function(dot(self.weights[i], input_data) + self.bias) for i in range(len(self))]

    def update_weights(self, weights: List[List[float]], bias: float = None) -> None:
        """
        Updates the weights and bias with a given set of weights.
        :param weights: The weights to change
        :param bias: The bias to be changed, optional.
        """
        assert len(self.weights) == len(weights), "Error in updating weights. The columns of the new weights do not match the current weight columns."
        assert len(self.weights[0]) == len(weights[0]), "Error in updating weights. The rows of the new weights do not match the current weight rows."
        
        self.weights = weights
        if bias: self.bias = bias

if __name__ == "__main__":
    example = Layer(nodes=3, prev_nodes=2)   
    example_prop = [random.uniform(-1.0, 1.0) for i in range(2)]

    print(example)
    print(example.forward_propagate(example_prop, sigmoid))

