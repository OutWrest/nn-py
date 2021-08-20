import random
from typing import Callable, List

class Layer:
    def __init__(self, nodes: int, prev_nodes: int) -> None:
        """
        Initializes a layer with a given number of nodes and a given number of weights.
        :param prev_weights: The number of weights in the previous layer.
        :param nodes: The number of nodes in the layer.
        """
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

    def propagate(self, input: List[float], activation_function: Callable) -> List[float]:
        """
        Propagates the values in the previous layer through the layer.
        :param input: The activations to carry through.
        :return: A list of the values in the layer.
        """ 
        return [activation_function(sum([i * j for i, j in zip(input, w)]) + self.bias) for w in self.weights]

if __name__ == "__main__":
    example = Layer(nodes=2, prev_nodes=2)
    print(example)