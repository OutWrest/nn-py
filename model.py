import math
from typing import List
from layer import Layer
from functools import reduce, total_ordering

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

class Model:
    def __init__(self) -> None:
        self.layers = []

    def addLayer(self, neurons: int, input_shape: int = None) -> None:
        if not self.layers:
            if not input_shape:
                raise Exception('Input shape not defined on first layer')
            self.layers.append( Layer(input_shape, neurons ))
        else:
            # Sets the input shape neurons from last layer
            self.layers.append( Layer(self.layers[-1].neurons, neurons ))
    
    def local_cost(self, output_activations: List[float], expected_activations: List[float]) -> float:
        assert len(output_activations) == len(expected_activations)
        
        loss = 0
        for predicted, expected in zip(output_activations, expected_activations):
            loss += pow(predicted-expected, 2)/2

        return loss
    
    def __cost(self, target: List[float], out: List[float]) -> List[float]:
        return [ pow(t_n - o_n, 2)/2 for t_n, o_n in zip(target, out) ]
    
    def __d_cost(self, target: List[float], out: List[float]) -> List[float]:
        return [ (t_n - o_n) for t_n, o_n in zip(target, out)]

    def __predict(self, activations: List[float]) -> List[List[float]]:
        ret = []

        for layer in self.layers:
            activations = layer.get_activations(activations, sigmoid)
            ret.append(activations)
        
        return ret
    
    def loadModel(self, weights: List[List[List[float]]], biases: List[List[float]]) -> None:
        for lweights, lbiases, layer in zip(weights, biases, self.layers):
            layer.load(lweights, lbiases)
            
    def predict(self, inputs: List[float]) -> List[float]:
        # There are layers that exist
        assert(self.layers) 
        # The first layer accepts input shape
        assert(self.layers[0].num_weights == len(inputs))

        for i, layer in enumerate(self.layers):
            inputs = layer.get_activations(inputs, sigmoid)
            print(f"[Layer {i}] {inputs}")

        return inputs
    
    def __backpropagate(self, activations, prev_activations, expdected: List[float], learning_rate: float) -> None:

        # parial derivative with respect to cost
        cost_t = self.__d_cost(activations[-1], expected)
        print(cost_t)
        
        # parial derivative with respect to activation function
        act_t = [ act*(1-act) for act in activations[-1] ]
        print(act_t)

        # partial derivative with respect to weights (everything)
        wei_t = [ cos_n * act_n * w_n for cos_n, act_n, w_n in zip(cost_t, act_t, activations[-2])]
        print(wei_t)

        # individal weight changes?
        n_wei = [[ weight - (learning_rate * weight_d) for weight in weights] for weights, weight_d in zip(weights_l, wei_t)]
        print(n_wei)
    
    def backpropagate(self, a_input: List[float], expected: List[float], learning_rate: float = 0.5) -> None:
        activations = self.__predict(a_input)
        weights = [[node.weights for node in layer.nodes] for layer in self.layers]     
        
        #for acts, weis in zip(activations, weis):
        #

if __name__ == "__main__":
    pass