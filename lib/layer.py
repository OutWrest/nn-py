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
        self.prev_nodes = prev_nodes
        self.weights = [[random.uniform(-1, 1) for _ in range(nodes)] for _ in range(prev_nodes)]
        self.nodes   = [random.uniform(-1, 1) for _ in range(nodes)]
        self.bias    = random.uniform(-1, 1)

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
        return len(self.nodes)

    def forwardpropagate(self, input_data: List[float], activation_function: Callable) -> List[float]:
        """
        Propagates the values in the previous layer through the layer. Handles laysrs with different number of nodes.
        :param input_data: The activations to carry through.
        :return: A list of the values in the layer.
        """ 
        assert len(input_data) == self.prev_nodes

        #print("Nodes", len(self))
        #print("Prev", self.prev_nodes)

        #print(reshape_list([[inp] * len(self) for inp in input_d]))
        #print(self.weights)

        #for i in range(len(self)):
        #    for j in range((self.prev_nodes)):
        #        print(reshape_list([[inp] * len(self) for inp in input_d])[i][j], reshape_list(self.weights)[i][j])
        #exit()
        ret = []

        for i in range(len(self)):
            ret.append(activation_function(sum(self.weights[j][i] * (input_data[j]) for j in range(self.prev_nodes)) + self.bias))
            #print([self.weights[j][i] * (input_data[j]) for j in range(self.prev_nodes)], self.bias)

        return ret

    def update_weights(self, weights: List[List[float]], bias: float = None) -> None:
        """
        Updates the weights and bias with a given set of weights. Includes error checking.
        :param weights: The weights to change
        :param bias: The bias to be changed, optional.
        """
        assert len(self.weights) == len(weights)
        assert len(self) == len(weights[0])
        
        for i in range(len(self)):
            for j in range(len(self.weights[0])):
                self.weights[i][j] -= weights[i][j]

        if bias: self.bias = bias

if __name__ == "__main__":
    example = Layer(nodes=2, prev_nodes=2)
    print(example)