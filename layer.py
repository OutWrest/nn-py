from typing import Callable, List
from node import Node

class Layer:
    def __init__(self, num_weights: int) -> None:
        self.nodes = [ Node(num_weights) for _ in range(num_weights) ]
    
    def get_activations(self, prev_activations: List[float], activation_function: Callable) -> List[float]:
        return [ n.get_activation(prev_activations, activation_function) for n in self.nodes ]

if __name__ == "__main__":
    pass