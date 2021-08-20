import math

from typing import List
from layer import Layer
from activation import sigmoid, d_sigmoid

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
    
    @staticmethod
    def t_error(expected: List[float], actual: List[float]) -> float:
        """
        Calculates the total error between the expected and actual values.
        :param expected: The expected values.
        :param actual: The actual values.
        :return: The error.
        """
        if len(expected) != len(actual):
            raise ValueError("Expected and actual data have different lengths.")

        return sum([0.5 * math.pow(expected[i] - actual[i], 2) for i in range(len(expected))])
    
    @staticmethod
    def l_error(expected: float, actual: float) -> float:
        """
        Calculates the error between the expected and actual values.
        :param expected: The expected value.
        :param actual: The actual value.
        :return: The error.
        """
        return 0.5 * math.pow(expected - actual, 2)

    def add(self, layer: Layer) -> None:
        """
        Adds a new layer to the model.
        :param layer: The layer to add.
        :return: None
        """
        if len(self.layers) == 0:
            assert self.input_shape is not None
            assert len(layer) == self.input_shape[0]
        else:
            assert len(layer.weights) == len(self.layers[-1])

        self.layers.append(layer)

    def forwardpropagate(self, input_data: list) -> list:
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
            output_data = layer.forwardpropagate(output_data, sigmoid)
        return output_data
    
    def fit(self, input_data: list, expected_data: list, learning_rate: float = 0.05, epochs: int = 100) -> None:
        """
        Trains the model using the given data.
        :param input_data: The input data.
        :param expected_data: The expected data.
        :param learning_rate: The learning rate.
        :param epochs: The number of epochs.
        :return: None
        """
        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")
        
        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        if len(expected_data) != self.output_shape[0]:
            raise ValueError("Expected data has invalid shape.")

        



if __name__ == "__main__":
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    # Forward propagation

    example = Model(input_shape=(2,))

    example.add(Layer(2, 2))
    example.add(Layer(2, 2))

    example.layers[0].weights = [
        [.15, .20], [.25, .30]
    ]
    example.layers[0].bias = .35

    example.layers[1].weights = [
        [.40, .45], [.50, .55]
    ]
    example.layers[1].bias = .60

    print(example.forwardpropagate([.05, .10]))

    # Backward propagation

