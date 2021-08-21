import math

from typing import Callable, List
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
    
    def __fpropagate(self, input_data: List[float], activation: Callable = sigmoid) -> List[float]:
        """
        Performs forward propagation on the model.
        :param input_data: The input data.
        :param activation: The activation function.
        :return: A list of activations.
        """
        ret = []

        if len(self.layers) == 0:
            raise ValueError("Model has no layers.")
        
        if len(input_data) != self.input_shape[0]:
            raise ValueError("Input data has invalid shape.")

        output_data = input_data
        for layer in self.layers:
            output_data = layer.forwardpropagate(output_data, activation)
            ret.append(output_data)

        return ret

    def forwardpropagate(self, input_data: List[float], activation: Callable = sigmoid) -> list:
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
            output_data = layer.forwardpropagate(output_data, activation)
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

        output_data = self.forwardpropagate(input_data, activation)
        return self.t_error(expected, output_data)

    def train(self, input_data: list, expected_data: list, learning_rate: float = 0.5, epochs: int = 100, activation: Callable = sigmoid) -> None:
        """
        Trains the model using the given data. Assumes a batch size of 1 for now.
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

        if len(expected_data) != len(self.layers[-1]):
            raise ValueError("Expected data has invalid shape.")

        weights_deltas = []

        output = self.forwardpropagate(input_data, activation)
        activations = self.__fpropagate(input_data, activation)

        d_error_wrt_out = [-(target-out) for target, out in zip(expected_data, output)]
        #print(d_error_wrt_out)
        d_out_wrt_error = [out * (1 - out) for out in output]
        #print(d_out_wrt_error)
        pd_out_wrt_w = [activations[-2][i] for i in range(len(d_out_wrt_error))]
        #print(pd_out_wrt_w)

        deltas = []
        for et, on, nw, weights in zip(d_error_wrt_out, d_out_wrt_error, pd_out_wrt_w, self.layers[-1].weights):
            delta = []
            for w in weights:
                change = w - (learning_rate * on * nw * et)
                delta.append(change)
                print(f"[Delta] {str(w): <4} - ({on: <20} * {nw: <20} * {et: <20}) = {change}")
            deltas.append(delta)
        weights_deltas.append(deltas)

        pd_e_wrt_out = [[e_o * o_n * self.layers[-1].weights[w][i] for e_o, o_n, w in zip(d_error_wrt_out, d_out_wrt_error, range(2))] for i in range(len(self.layers[-1].weights[0]))]

        e_total_out = [sum(e_wrt_out) for e_wrt_out in pd_e_wrt_out]
        d_pd_out_wrt_w = [act * (1 - act) for act in pd_out_wrt_w]
        
        deltas = []
        for et, inp, ow, weights in zip(e_total_out, input_data, d_pd_out_wrt_w, self.layers[0].weights):
            delta = []
            for w in weights:
                change = w - (learning_rate * inp * ow * et)
                delta.append(change)
                print(f"[Delta] {str(w): <4} - ({inp: <20} * {ow: <20} * {et: <20}) = {change}")
            deltas.append(delta)
        weights_deltas.append(deltas)

        # Change weights
        for layer, deltas in zip(self.layers, weights_deltas):
            layer.weights = [w - d for w, d in zip(layer.weights, deltas)]

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

    # print(example.forwardpropagate([.05, .10]))
    print(f"Total Error: {example.get_error([.05, .10], [.01, .99])}")

    # Backward propagation

    example.train([.05, .10], [.01, .99])

