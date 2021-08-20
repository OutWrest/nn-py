from typing import List, Callable
import random

class Node:
    def __init__(self, num_weights: int, weights: List[float] = None) -> None:
        if weights:
            self.weights = weights
        else:
            self.weights = [ (random.random() * 2) - 1 for _ in range(num_weights) ]
    
    def __repr__(self) -> str:
        return f'Node({self.weights})'
    
    def get_activation(self, activations: List[float]) -> float:
        return sum([x * w for x, w in zip(activations, self.weights)])

if __name__ == "__main__":
    pass