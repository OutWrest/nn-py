from typing import Callable, List
from node import Node

class Layer:
    def __init__(self, num_weights: int, neurons: int, weights: List[List[float]] = None, biases: List[float] = None) -> None:
        self.neurons = neurons
        self.num_weights = num_weights

        if weights and biases:
            self.nones = [ Node(num_weights, weight, bias) for weight, bias in zip(weights, biases) ]
        else:
            self.nodes = [ Node(num_weights) for _ in range(neurons) ]
    
    def get_activations(self, prev_activations: List[float], activation_function: Callable) -> List[float]:
        return [ n.get_activation(prev_activations, activation_function) for n in self.nodes ]

if __name__ == "__main__":
    pass