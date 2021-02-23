from typing import List, Callable
import random

class Node:
    def __init__(self, num_weights: int) -> None:
        self.bias = (random.random() * 2) - 1
        self.weights = [ (random.random() * 2) - 1 for _ in range(num_weights) ]
    
    def get_activation(self, activations: List[float], activation_function: Callable) -> float:
        return activation_function(sum([x * w for x, w in zip(activations, self.weights)]) + self.bias)

if __name__ == "__main__":
    pass