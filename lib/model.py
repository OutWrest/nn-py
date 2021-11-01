import math

from typing import Callable, List

try:
    from lib.layer import Layer
    from lib.activation import *
    from lib.helpers import *
except ModuleNotFoundError:
    from layer import Layer
    from activation import *
    from helpers import *

class Model:
    def __init__(self, input_shape: tuple) -> None:
        """
        Initializes a new instance of the Model class.
        :param input_shape: The shape of the input data.
        :return: None
        """
        self.layers = []
        self.input_shape = input_shape
        self.output_shape = None
    
    def __fpropagate(self, input_data: List[float], activation: Callable = sigmoid) -> List[float]:
        """
        Performs forward propagation on the model.
        :param input_data: The input data.
        :param activation: The activation function.
        :return: A list of activations and output.
        """
        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")
        
        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        ret = []
        ret.append(input_data)

        output_data = input_data
        for layer in self.layers:
            output_data = layer.forward_propagate(output_data, activation)
            ret.append(output_data)

        return ret

    def train_delta(self, input_data: list, expected_data: list, activation: Callable = sigmoid, d_activiation: Callable = d_sigmoid) -> None:
        """
        Trains the model using the given data. Assumes a batch size of 1 for now.
        :param input_data: The input data.
        :param expected_data: The expected data.
        :param learning_rate: The learning rate.
        :param epochs: The number of epochs.
        :return: The weight deltas
        """
        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")
        
        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        if len(expected_data) != len(self.layers[-1]):
            raise ValueError("Expected data has invalid shape.")

        activations = self.__fpropagate(input_data, activation)

        t_error_arr = [out - target for out, target in zip(activations[-1], expected_data)]
        total_deltas = [[error * d_activiation(out) for error, out in zip(t_error_arr, activations[-1])]]

        #for i in range(len(self.layers) - 2, 1, -1):
        #    transposed = reshape_list(self.layers[i - 1].weights)
        #    prev_error = [dot(total_deltas[-1], transposed[j]) for j in range(self.layers[i - 1].prev_nodes)]
        #    l_delta = [error * activation for error, activation in zip(prev_error, activations[i])]
        #    total_deltas.append(l_delta)

        for i in range(len(self.layers) - 2, 0, -1):
            transposed = reshape_list(self.layers[i + 1].weights)
            error = [dot(total_deltas[-1], transposed[j]) for j in range(self.layers[i - 1].prev_nodes)]
            grad = [activations[i - 1]]
            print(grad, error)
            exit()

        #total_deltas = total_deltas[::-1]


        return total_deltas

    def fit(self):
        pass

    def add(self, layer: Layer) -> None:
        """
        Adds a new layer to the model.
        :param layer: The layer to add.
        :return: None
        """
        if len(self.layers) == 0:
            assert self.input_shape is not None
            assert layer.prev_nodes == self.input_shape[0]
        else:
            assert layer.prev_nodes == len(self.layers[-1])

        self.layers.append(layer)

    def predict(self, input_data: List[float], activation: Callable = sigmoid) -> list:
        """
        Propagates the input data through the model.
        :param input_data: The input data.
        :return: The output data through the model.
        """

        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")

        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        output_data = input_data
        for layer in self.layers:
            output_data = layer.forward_propagate(output_data, activation)
        return output_data
    
    def get_error(self, input_data: List[float], expected: List[float], activation: Callable = sigmoid) -> float:
        """
        Calculates the error of the model of a specific input_data.
        :param input_data: The input data.
        :param expected: The expected output data.
        :param activation: The activation function.
        :return: The error.
        """
        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")
        
        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        if len(expected) != len(self.layers[-1]):
            raise ValueError("Expected output data has invalid shape.")

        output_data = self.predict(input_data, activation)
        return t_error(expected, output_data)
        

if __name__ == "__main__":
    example = Model(input_shape=(2,))

    example.add(Layer(2, 2))
    example.add(Layer(2, 2))

    example.layers[0].update_weights(
        [[.15, .20], [.25, .30]], .35
    )

    example.layers[1].update_weights(
        [[.40, .45], [.50, .55]], .60
    )

    print("Forward Prop:", example.predict([.05, .10]))
    
    wd = example.train_delta([.05, .10], [.01, .99])
    print(wd)