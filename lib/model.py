import math

from typing import Callable, List

try:
    from lib.layer import Layer
    from lib.activation import sigmoid, d_sigmoid
except ModuleNotFoundError:
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
    def d_sigmoid(x: float) -> float:
        """
        Calculates the derivative of the sigmoid function for a given input that has already been passed through sigmoid.
        :param x: The input.
        :return: The derivative.
        """
        return x * (1 - x)
    
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

    @staticmethod
    def reshape_list(data: List[List[float]]) -> List[float]:
        """
        Reshapes a x by y list of lists of floats into a y by x list of lists of floats.
        :param data: The list of lists.
        :return: The reshaped list.
        """
        ret = []
        for i in range(len(data[0])):
            ret.append([])
            for j in range(len(data)):
                ret[i].append(data[j][i])
        return ret


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
            assert layer.prev_nodes == len(self.layers[-1])

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
        d_out_wrt_error = [self.d_sigmoid(out) for out in output]
        #print(d_out_wrt_error)
        pd_out_wrt_w = activations[-2]
        #print(pd_out_wrt_w)

        deltas = [[0] * len(self.layers[-1].weights) for _ in range(len(self.layers[-1]))]
        print(deltas)
        for nw, i in zip(pd_out_wrt_w, range(len(self.layers[-1].weights))):
            for j in range(len(self.layers[-1])):
                change = self.reshape_list(self.layers[-1].weights)[j][i] - (learning_rate * d_error_wrt_out[j] * nw * d_out_wrt_error[j])
                deltas[j][i] = change
                #print(f"[Delta] {str(self.layers[-1].weights[j][i]): <4} - ({ d_error_wrt_out[j]: <20} * {nw: <20} * {d_out_wrt_error[j]: <20}) = {change}")
        weights_deltas.append(deltas)

        + = [[e_o * o_n * self.reshape_list(self.layers[-1].weights)[w][i] for e_o, o_n, w in zip(d_error_wrt_out, d_out_wrt_error, range(len(self.layers[-2])))] for i in range(len(self.layers[-1].weights[0]))]

        e_total_out = [sum(eout) for eout in pd_e_wrt_out]
        d_pd_out_wrt_w = [self.d_sigmoid(act) for act in pd_out_wrt_w]
        
        deltas = [[0] * len(self.layers[-2]) for _ in range(len(self.layers[-2].weights))]
        for et, inp, ow, i in zip(e_total_out, input_data, d_pd_out_wrt_w, range(len(self.layers[0].weights))):
            for j in range(len(self.layers[0])):
                change = self.layers[0].weights[j][i] - (learning_rate * inp * ow * et)
                deltas[j][i] = change
                #print(f"[Delta] {str(self.layers[0].weights[j][i]): <4} - ({learning_rate} * {inp: <20} * {ow: <20} * {et: <20}) = {change}")
        weights_deltas.append(deltas)

        # Change weights

        weights_deltas = ([self.reshape_list(weights) for weights in weights_deltas[::-1]])

        for weight, layer in zip(weights_deltas, self.layers):
            layer.update_weights(weight)

if __name__ == "__main__":
    # https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/

    # Forward propagation

    example = Model(input_shape=(2,))

    INIT_WEIGHTS, TEST = False, False

    example.add(Layer(2, 2))
    example.add(Layer(3, 2))

    if INIT_WEIGHTS:
        example.layers[0].weights = [
            [.15, .25], [.20, .30]
        ]
        example.layers[0].bias = .35

        example.layers[1].weights = [
            [.40, .50], [.45, .55]
        ]
        example.layers[1].bias = .60

        if TEST:
            example.layers[1].weights = [
                [.40, .45, .55], [.60, .65, .70]
            ]

    print(example.forwardpropagate([.05, .10]))
    print(f"Total Error: {example.get_error([.05, .10], [.01, .99, .05])}")

    # Backward propagation

    BACK_PROP = True
    
    if BACK_PROP:
        for k in range(10000):
            example.train([.05, .10], [.01, .99, .05])
    else:
        example.train([.05, .10], [.01, .99, .05])

    print(f"Total Error: {example.get_error([.05, .10], [.01, .99, .05])}")
  