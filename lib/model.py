import math
from tqdm import tqdm

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
    
    def __fpropagate(self, input_data: List[float], activation: Callable = sigmoid) -> List[float]:
        """
        Performs forward propagation on the model.
        :param input_data: The input data.
        :param activation: The activation function.
        :return: A list of activations and output.
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

        return ret, output_data

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
        return t_error(expected, output_data)
    
    def train_delta(self, input_data: list, expected_data: list, learning_rate: float = 0.5, activation: Callable = sigmoid) -> None:
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

        weights_deltas = []

        activations, outputs = self.__fpropagate(input_data, activation)

        d_error_wrt_out = [-(target-out) for target, out in zip(expected_data, outputs)]
        #print(d_error_wrt_out)
        d_out_wrt_error = [d_sigmoid(out) for out in outputs]
        #print(d_out_wrt_error)
        pd_out_wrt_w = [[act] * len(outputs) for act in activations[-2]]
        #print(pd_out_wrt_w)

        weights = []
        for i, weights_p in enumerate(self.layers[-1].weights):
            weight = []
            for k, w in enumerate(weights_p):
                weight.append((learning_rate * pd_out_wrt_w[i][k] * d_out_wrt_error[k] * d_error_wrt_out[k]))
                #print(f"w_d = {w} - ({learning_rate} * {pd_out_wrt_w[k][i]} * {d_out_wrt_error[k]} {d_error_wrt_out[k]})")
            weights.append(weight)
        weights_deltas.append(weights)

        pd_e_wrt_out = [sum([w * d_error_wrt_out[j] * d_out_wrt_error[j] for j, w in enumerate(weights_p)]) for weights_p in self.layers[-1].weights]
        #print(pd_e_wrt_out)
        pd_sig_wrt_out = [d_sigmoid(act) for act in activations[-2]]
        #print(pd_sig_wrt_out)

        weights = []
        for i, weights_p, inp_p in zip(range(len(self.layers[-2].weights)), self.layers[-2].weights, input_data[::-1]):
            weight = []
            for k, w in enumerate(weights_p):
                weight.append((learning_rate * pd_e_wrt_out[k] * (inp_p) * pd_sig_wrt_out[k]))
                #print(f"w_d = {w} - ({learning_rate} * {pd_e_wrt_out[k]} * {inp_p} * {pd_sig_wrt_out[k]})")
            weights.append(weight)
        weights_deltas.append(weights)

        return weights_deltas, outputs

    def train_batch(self, input_data: List[List[float]], expected_data: List[List[float]], batch_size: int = 32, learning_rate: float = 0.5, activation: Callable = sigmoid) -> None:
        """
        Trains the model using the given data. Assumes a batch size of 32 unless given.
        :param input_data: The input data.
        :param expected_data: The expected data.
        :param batch_size: The batch size.
        :param learning_rate: The learning rate.
        :param epochs: The number of epochs.
        :return: None
        """
        if batch_size == 0:
            batch_size = len(input_data)
        
        for i in tqdm(range(0, len(input_data), batch_size)):
            weights_total = [[[0 for _ in range(len(weights_p))] for weights_p in self.layers[-1].weights], [[0 for _ in range(len(weights_p))] for weights_p in self.layers[-2].weights]]
            batch_error_average = 0
            
            total = []
            for b in range(batch_size):
                try:
                    weights_deltas, outputs = self.train_delta(input_data[i + b], expected_data[i + b], learning_rate, activation)

                    if batch_error_average == 0:
                        batch_error_average = t_error(expected_data[i + b], outputs)
                    else:
                        batch_error_average = (batch_error_average * (i + b) + t_error(expected_data[i + b], outputs)) / (i + b + 1)

                except IndexError:
                    if total: 
                        batch_size = len(total)
                        batch_error_average = sum(total) / batch_size
                        break
                    return
                total.append(weights_deltas)

            for k_t in total:
                for i in range(len(weights_total)):
                    for j in range(len(weights_total[i])):
                        for k in range(len(weights_total[i][j])):
                            weights_total[i][j][k] += k_t[i][j][k] / batch_size

            for weight, layer in zip(weights_total[::-1], self.layers):
                layer.update_weights(weight)

    def train(self, input_data: list, expected_data: list, learning_rate: float = 0.5, activation: Callable = sigmoid) -> None:
        """
        Trains the model using the given data. Assumes a batch size of 1 for now.
        :param input_data: The input data.
        :param expected_data: The expected data.
        :param learning_rate: The learning rate.
        :param epochs: The number of epochs.
        :return: None
        """
        weights_deltas, _ = self.train_delta(input_data, expected_data, learning_rate, activation)

        for weight, layer in zip(weights_deltas[::-1], self.layers):
            layer.update_weights(weight)

if __name__ == "__main__":
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    # Forward propagation

    example = Model(input_shape=(2,))

    INIT_WEIGHTS = True

    example.add(Layer(2, 2))
    example.add(Layer(2, 2))

    prop, out = [.05, .10] , [.01, .99]

    if INIT_WEIGHTS:
        example.layers[0].weights = [
            [.15, .25], [.20, .30]
        ]
        example.layers[0].bias = .35

        example.layers[1].weights = [
            [.40, .50], [.45, .55]
        ]
        example.layers[1].bias = .60

    # Backward propagation

    print(f"Test prop: {example.forwardpropagate(prop)}")

    print(f"Total Error: {example.get_error(prop, out)}")

    prop, out = [[.05, .10], [.05, .10]], [[.01, .99], [.01, .99]]

    BACK_PROP = False
    
    example.train_batch(prop, out, batch_size=2)

    prop, out = [.05, .10] , [.01, .99]

    print(f"Test prop: {example.forwardpropagate(prop)}")

    print(f"Total Error: {example.get_error(prop, out)}")

    #print(example.layers[1].weights)
  