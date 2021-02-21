from typing import List, Callable

class Node:
    def __init__(self, num_weights: int) -> None:
        self.bias = 0
        self.weights = [ 0 for _ in range(num_weights) ]
    
    def get_activation(self, activations: List[float], activation_function: Callable) -> float:
        return activation_function(sum([x * w for x, w in zip(activations, self.weights)]) + self.bias)

if __name__ == "__main__":
    pass