import random
from typing import Callable, List
from node import Node

class Layer:
    def __init__(self, num_weights: int, neurons: int, weights: List[List[float]] = None, bias: float = None) -> None:
        self.neurons = neurons
        self.num_weights = num_weights
        self.bias = bias if bias else (random.random() * 2) - 1

        if weights:
            self.nodes = [ Node(num_weights, weight) for weight in weights ]
        else:
            self.nodes = [ Node(num_weights) for _ in range(neurons) ]
        
    def __repr__(self) -> str:
        return f"Layer({self.nodes})"
    
    def load(self, weights: List[List[float]], bias: float) -> None:
        self.bias = bias
        for nweights, node in zip(weights, self.nodes):
            node.weights = nweights
    
    def get_weights(self) -> List[List[float]]:
        return [ node.weights for node in self.nodes ]
    
    def get_activations(self, prev_activations: List[float], activation_function: Callable) -> List[float]:
        return [ activation_function(n.get_activation(prev_activations) + self.bias) for n in self.nodes ]

if __name__ == "__main__":
    pass